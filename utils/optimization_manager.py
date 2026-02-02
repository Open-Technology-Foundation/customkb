"""
Optimization manager for CustomKB performance tuning.

This module provides functionality to analyze and optimize CustomKB configurations
based on system resources and knowledgebase characteristics.
"""

import configparser
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.index_manager import create_missing_indexes
from utils.enums import OptimizationTier
from utils.logging_config import get_logger

logger = get_logger(__name__)


def get_system_memory_gb() -> float:
  """Get total system memory in GB."""
  try:
    import psutil
    return psutil.virtual_memory().total / (1024**3)
  except (ImportError, AttributeError) as e:
    # Fallback to reading /proc/meminfo if psutil is not available
    logger.debug(f"psutil not available, falling back to /proc/meminfo: {e}")
    try:
      with open('/proc/meminfo') as f:
        for line in f:
          if line.startswith('MemTotal:'):
            kb = int(line.split()[1])
            return kb / (1024**2)
    except (OSError, FileNotFoundError, ValueError) as e:
      logger.warning(f"Could not determine system memory: {e}")
      return 16  # Safe default


def get_optimized_settings(memory_gb: float = None) -> dict:
  """
  Get optimized settings based on available system memory and GPU.

  Memory tiers:
  - Low: < 16GB (conservative settings)
  - Medium: 16-64GB (balanced performance)
  - High: 64-128GB (high performance)
  - Very High: > 128GB (maximum performance)

  Args:
      memory_gb: System memory in GB. If None, auto-detect.

  Returns:
      Dict containing tier info, optimization settings, and GPU info.
  """
  if memory_gb is None:
    memory_gb = get_system_memory_gb()

  # Get GPU information
  from utils.gpu_utils import get_gpu_memory_mb, get_safe_gpu_memory_limit_mb
  gpu_memory_mb = get_gpu_memory_mb()
  gpu_info = {}

  if gpu_memory_mb:
    gpu_info = {
      'gpu_memory_mb': gpu_memory_mb,
      'gpu_memory_gb': gpu_memory_mb / 1024.0,
      'safe_limit_mb': get_safe_gpu_memory_limit_mb(buffer_gb=4.0),
      'detected': True
    }
  else:
    gpu_info = {'detected': False}
    gpu_memory_mb = 0

  # Determine optimization tier based on memory
  tier_enum = OptimizationTier.from_memory(memory_gb)
  tier = tier_enum.value

  # Base settings that scale with memory
  match tier_enum:
    case OptimizationTier.LOW:
      # Low memory settings (conservative)
      memory_factor = 0.25
      thread_factor = 0.5
      batch_factor = 0.5
    case OptimizationTier.MEDIUM:
      # Medium memory settings (balanced)
      memory_factor = 0.5
      thread_factor = 0.75
      batch_factor = 0.75
    case OptimizationTier.HIGH:
      # High memory settings (performance)
      memory_factor = 0.75
      thread_factor = 1.0
      batch_factor = 1.0
    case OptimizationTier.VERY_HIGH:
      # Very high memory settings (maximum performance)
      memory_factor = 1.0
      thread_factor = 1.5
      batch_factor = 1.5

  # Calculate scaled values
  # Reduced base memory cache from 500000 to 200000 to prevent OOM
  memory_cache = int(200000 * memory_factor)
  io_threads = max(4, int(32 * thread_factor))
  cache_threads = max(4, int(16 * thread_factor))
  api_concurrency = max(8, int(32 * thread_factor))
  api_min_concurrency = max(3, int(16 * thread_factor))
  # Reduced base embedding batch from 1000 to 750 for stability
  embedding_batch = int(750 * batch_factor)
  # Conservative batch sizes to prevent memory exhaustion
  file_batch = int(2000 * batch_factor)  # Reduced from 5000
  sql_batch = int(2000 * batch_factor)   # Reduced from 5000
  reference_batch = max(5, int(30 * batch_factor))  # Reduced from 50
  # Conservative reranking batch size
  reranking_batch = int(32 * batch_factor)
  reranking_cache = int(10000 * memory_factor)

  return {
    'tier': tier,
    'memory_gb': memory_gb,
    'optimizations': {
      'DEFAULT': {
        'query_top_k': '30',
      },
      'API': {
        'api_call_delay_seconds': '0.01',
        'api_max_concurrency': str(api_concurrency),
        'api_min_concurrency': str(api_min_concurrency),
      },
      'LIMITS': {
        'max_file_size_mb': str(int(500 * batch_factor)),
        'max_query_file_size_mb': str(max(1, int(10 * batch_factor))),
        'memory_cache_size': str(memory_cache),
        'cache_memory_limit_mb': str(int(500 * memory_factor)),  # Memory-based cache limit
        'max_query_length': str(int(20000 * batch_factor)),  # Reduced from 50000
        'max_config_value_length': str(int(5000 * batch_factor)),
        'max_json_size': str(int(50000 * batch_factor)),
      },
      'PERFORMANCE': {
        'embedding_batch_size': str(embedding_batch),
        'checkpoint_interval': str(max(10, int(30 * batch_factor))),  # Reduced
        'commit_frequency': str(int(3000 * batch_factor)),  # Reduced from 5000
        'io_thread_pool_size': str(io_threads),
        'cache_thread_pool_size': str(cache_threads),
        'file_processing_batch_size': str(file_batch),
        'sql_batch_size': str(sql_batch),
        'reference_batch_size': str(reference_batch),
        'query_cache_ttl_days': '30',
      },
      'ALGORITHMS': {
        'small_dataset_threshold': str(int(10000 * batch_factor)),
        'medium_dataset_threshold': str(int(1000000 * batch_factor)),
        'ivf_centroid_multiplier': str(max(4, int(8 * batch_factor))),
        'max_centroids': str(int(1024 * batch_factor)),
        'token_estimation_sample_size': str(int(50 * batch_factor)),
        'max_chunk_overlap': str(int(200 * batch_factor)),
        'heading_search_limit': str(int(500 * batch_factor)),
        'entity_extraction_limit': str(int(1000 * batch_factor)),
        # Hybrid search configuration
        # Disable hybrid search for systems under 64GB to prevent crashes
        'enable_hybrid_search': 'true' if memory_gb >= 16 else 'false',
        # RRF is more robust than weighted averaging for score fusion
        'hybrid_fusion_method': 'rrf',
        'rrf_k': '60',  # Standard RRF ranking constant
        'vector_weight': '0.7',  # Used with 'weighted' fusion method
        'bm25_weight': '0.3',    # Used with 'weighted' fusion method
        'bm25_rebuild_threshold': str(int(5000 * batch_factor)),
        # Conservative BM25 limit to prevent memory exhaustion
        'bm25_max_results': str(min(750, int(1000 * memory_factor))),
        # Query enhancement settings
        'enable_query_enhancement': 'true',
        'query_enhancement_synonyms': 'false',  # Disabled - can harm precision in technical content
        'query_enhancement_spelling': 'true',   # Keep spelling correction enabled
        'max_synonyms_per_word': str(max(1, int(3 * memory_factor))),
        'query_enhancement_cache_ttl_days': '60',
        'reranking_top_k': str(min(30, int(30 * batch_factor))),  # Capped at 30
        'reranking_batch_size': str(reranking_batch),
        # Always use CPU for reranking to avoid GPU memory conflicts
        'reranking_device': 'cpu',
        'reranking_cache_size': str(reranking_cache),
        # GPU optimization for FAISS - dynamic based on GPU memory
        'faiss_gpu_batch_size': str(get_gpu_batch_size(gpu_memory_mb, batch_factor)),
        # Use float16 based on GPU memory availability
        'faiss_nprobe': '32',
        'faiss_gpu_use_float16': 'true' if gpu_memory_mb < 24576 else 'false',
        # GPU memory buffer for safety
        'faiss_gpu_memory_buffer_gb': '4.0',
      }
    },
    'gpu_info': gpu_info
  }


def get_gpu_batch_size(gpu_memory_mb: int, batch_factor: float) -> int:
  """Calculate appropriate GPU batch size based on GPU memory."""
  if gpu_memory_mb == 0:
    return 1024  # Default for no GPU
  elif gpu_memory_mb < 8192:  # < 8GB
    return 512
  elif gpu_memory_mb < 16384:  # < 16GB
    return 1024
  elif gpu_memory_mb < 24576:  # < 24GB
    return int(2048 * batch_factor)
  else:  # >= 24GB
    return int(4096 * batch_factor)


def find_kb_configs(vectordbs_dir: str) -> list[str]:
  """Find all knowledgebase configuration files in VECTORDBS."""
  configs = []

  for root, dirs, files in os.walk(vectordbs_dir):
    # Skip hidden directories and virtual environments
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['.venv', 'venv', '__pycache__']]

    for file in files:
      if file.endswith('.cfg'):
        config_path = os.path.join(root, file)
        # Skip backup files and pyvenv.cfg files
        if ('backup' not in config_path and
            'optimized' not in config_path and
            file != 'pyvenv.cfg'):
          configs.append(config_path)

  return sorted(configs)


def backup_config(config_path: str) -> str:
  """Create a backup of the configuration file."""
  backup_dir = os.path.join(os.path.dirname(config_path), 'backups')
  os.makedirs(backup_dir, exist_ok=True)

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  backup_name = f"{os.path.basename(config_path)}.{timestamp}.bak"
  backup_path = os.path.join(backup_dir, backup_name)

  shutil.copy2(config_path, backup_path)
  logger.info(f"Created backup: {backup_path}")

  return backup_path


def optimize_config(config_path: str, dry_run: bool = False, memory_gb: float = None, check_indexes: bool = True) -> dict:
  """Apply performance optimizations to a configuration file."""
  logger.info(f"Processing: {config_path}")

  # Check FAISS index size
  kb_dir = os.path.dirname(config_path)
  kb_name = os.path.basename(kb_dir)
  faiss_file = os.path.join(kb_dir, f"{kb_name}.faiss")
  faiss_size_mb = 0
  if os.path.exists(faiss_file):
    faiss_size_mb = os.path.getsize(faiss_file) / (1024 * 1024)

  # Get optimized settings based on system memory
  settings_info = get_optimized_settings(memory_gb)
  performance_optimizations = settings_info['optimizations']
  gpu_info = settings_info.get('gpu_info', {})

  # Adjust GPU settings based on FAISS index size and GPU memory
  if gpu_info and faiss_size_mb > 0:
    from utils.gpu_utils import should_use_gpu_for_index
    gpu_suitable, reason = should_use_gpu_for_index(faiss_size_mb, kb_config=None)

    if not gpu_suitable:
      logger.info(f"GPU not suitable for FAISS index: {reason}")
      # Conservative GPU settings for large indexes
      performance_optimizations['ALGORITHMS']['faiss_gpu_batch_size'] = '512'  # Minimal batch size
      performance_optimizations['ALGORITHMS']['faiss_gpu_use_float16'] = 'true'  # Always use float16
    else:
      logger.info(f"GPU suitable for FAISS index: {reason}")

  # Load existing configuration
  config = configparser.ConfigParser()
  config.read(config_path)

  # Track changes
  changes = {}

  # Apply optimizations
  for section, settings in performance_optimizations.items():
    if section not in config and section != 'DEFAULT':
      config.add_section(section)

    for key, new_value in settings.items():
      old_value = config.get(section, key, fallback=None)

      # Only update if different
      if old_value != new_value:
        if not dry_run:
          config.set(section, key, new_value)

        if section not in changes:
          changes[section] = {}
        changes[section][key] = {
          'old': old_value,
          'new': new_value
        }

  # Save optimized configuration
  if not dry_run and changes:
    # Create backup first
    backup_path = backup_config(config_path)

    # Write optimized config
    with open(config_path, 'w') as f:
      config.write(f)

    logger.info(f"Updated configuration: {config_path}")
    logger.info(f"Backup saved to: {backup_path}")

  # Check and create missing indexes if requested
  if check_indexes:
    try:
      kb = KnowledgeBase(config_path)
      db_path = kb.knowledge_base_db

      if Path(db_path).exists():
        logger.info("Checking database indexes...")
        missing_indexes = create_missing_indexes(db_path, dry_run=dry_run)
        if missing_indexes:
          if 'indexes' not in changes:
            changes['indexes'] = {}
          for idx in missing_indexes:
            changes['indexes'][idx] = {
              'old': 'missing',
              'new': 'created' if not dry_run else 'would create'
            }
    except (FileNotFoundError, sqlite3.Error, OSError) as e:
      logger.warning(f"Could not check indexes: {e}")

  return changes


def analyze_kb_size(config_path: str) -> dict:
  """Analyze the size of a knowledgebase."""
  kb_dir = os.path.dirname(config_path)
  stats = {
    'config_path': config_path,
    'kb_name': os.path.basename(kb_dir),
    'db_size_mb': 0,
    'vector_size_mb': 0,
    'total_size_mb': 0
  }

  # Check for database file
  db_files = [f for f in os.listdir(kb_dir) if f.endswith('.db')]
  if db_files:
    db_path = os.path.join(kb_dir, db_files[0])
    if os.path.exists(db_path):
      stats['db_size_mb'] = os.path.getsize(db_path) / 1024 / 1024

  # Check for vector file
  vector_files = [f for f in os.listdir(kb_dir) if f.endswith('.faiss')]
  if vector_files:
    vector_path = os.path.join(kb_dir, vector_files[0])
    if os.path.exists(vector_path):
      stats['vector_size_mb'] = os.path.getsize(vector_path) / 1024 / 1024

  stats['total_size_mb'] = stats['db_size_mb'] + stats['vector_size_mb']

  return stats


def show_optimization_tiers() -> str:
  """
  Display optimization settings for all memory tiers.

  Returns:
      Formatted string showing tier settings
  """
  output = ["Optimization Tiers for CustomKB", "=" * 80, ""]

  # Show current GPU info if available
  from utils.gpu_utils import get_gpu_info_string
  gpu_info_str = get_gpu_info_string()
  output.extend([
    "System Configuration:",
    f"  {gpu_info_str}",
    ""
  ])

  tiers = [
    (8, "Low Memory (<16GB)"),
    (32, "Medium Memory (16-64GB)"),
    (96, "High Memory (64-128GB)"),
    (256, "Very High Memory (>128GB)")
  ]

  for memory_gb, tier_name in tiers:
    settings = get_optimized_settings(memory_gb)
    opts = settings['optimizations']

    output.extend([
      f"{tier_name} - Example: {memory_gb}GB System",
      "-" * 80,
      f"Tier: {settings['tier'].upper()}",
      "",
      "Key Settings:",
      f"  Memory cache size: {opts['LIMITS']['memory_cache_size']}",
      f"  Embedding batch size: {opts['PERFORMANCE']['embedding_batch_size']}",
      f"  Reference batch size: {opts['PERFORMANCE']['reference_batch_size']}",
      f"  IO thread pool: {opts['PERFORMANCE']['io_thread_pool_size']}",
      f"  Cache thread pool: {opts['PERFORMANCE']['cache_thread_pool_size']}",
      f"  API max concurrency: {opts['API']['api_max_concurrency']}",
      f"  BM25 max results: {opts['ALGORITHMS']['bm25_max_results']}",
      f"  Hybrid search enabled: {opts['ALGORITHMS']['enable_hybrid_search']}",
      f"  Reranking batch size: {opts['ALGORITHMS']['reranking_batch_size']}",
      f"  GPU batch size: {opts['ALGORITHMS']['faiss_gpu_batch_size']}",
      f"  GPU float16 mode: {opts['ALGORITHMS']['faiss_gpu_use_float16']}",
      ""
    ])

  output.extend([
    "Notes:",
    "- Actual settings scale based on exact memory amount",
    "- GPU settings adjust automatically based on FAISS index size",
    "- Use --memory-gb flag to override auto-detection"
  ])

  return '\n'.join(output)


def format_changes(config_path: str, changes: dict) -> str:
  """Format configuration changes for display."""
  if not changes:
    return f"{config_path}: No changes needed (already optimized)"

  output = [f"\n{config_path}:", "-" * 80]

  for section, settings in changes.items():
    output.append(f"\n[{section}]")
    for key, values in settings.items():
      old = values['old'] or '<not set>'
      new = values['new']
      output.append(f"  {key}: {old} -> {new}")

  return '\n'.join(output)


def process_optimize(args, logger) -> str:
  """
  Process the optimize command for CustomKB performance tuning.

  This command provides several optimization features:
  - Analyzes system resources and applies tier-based configuration optimizations
  - Creates missing database indexes for improved query performance
  - Shows KB size analysis and recommendations
  - Displays optimization tier settings

  Args:
      args: Command line arguments containing:
          - target: KB config file, directory, or name (optional)
          - dry_run: Show changes without applying them
          - analyze: Show KB size analysis
          - show_tiers: Display all optimization tiers
          - memory_gb: Override system memory detection
      logger: Logger instance

  Returns:
      Formatted result message string with optimization details
  """
  # Handle --show-tiers flag
  if hasattr(args, 'show_tiers') and args.show_tiers:
    return show_optimization_tiers()

  # Find configurations to process
  vectordbs_dir = os.getenv('VECTORDBS', '/var/lib/vectordbs')

  # Debug logging
  logger.debug(f"VECTORDBS directory: {vectordbs_dir}")
  logger.debug(f"Target: {args.target if hasattr(args, 'target') else 'None'}")

  if args.target:
    logger.debug(f"Checking target: {args.target}")

    # First, try to find it as a KB name in VECTORDBS (most common case)
    # This avoids conflicts with local directories that have the same name
    if not os.path.isabs(args.target) and not args.target.endswith('.cfg'):
      logger.debug("Checking if target is a KB name in VECTORDBS")
      potential_path = get_fq_cfg_filename(args.target)
      logger.debug(f"get_fq_cfg_filename returned: {potential_path}")
      if potential_path and os.path.exists(potential_path):
        configs = [potential_path]
      else:
        # Fall back to checking if it's a local file or directory
        logger.debug("Not found as KB name, checking local paths")
        logger.debug(f"Is file? {os.path.isfile(args.target)}")
        logger.debug(f"Is dir? {os.path.isdir(args.target)}")

        if os.path.isfile(args.target) and args.target.endswith('.cfg'):
          configs = [args.target]
        elif os.path.isdir(args.target):
          configs = find_kb_configs(args.target)
        else:
          logger.error(f"Cannot find configuration: {args.target}")
          return f"Error: Cannot find configuration: {args.target}"
    else:
      # Absolute path or .cfg file - check normally
      logger.debug(f"Is file? {os.path.isfile(args.target)}")
      logger.debug(f"Is dir? {os.path.isdir(args.target)}")

      if os.path.isfile(args.target) and args.target.endswith('.cfg'):
        configs = [args.target]
      elif os.path.isdir(args.target):
        configs = find_kb_configs(args.target)
      else:
        logger.error(f"Cannot find configuration: {args.target}")
        return f"Error: Cannot find configuration: {args.target}"
  else:
    # Check if VECTORDBS directory exists
    if not os.path.exists(vectordbs_dir):
      logger.error(f"VECTORDBS directory does not exist: {vectordbs_dir}")
      return f"Error: VECTORDBS directory does not exist: {vectordbs_dir}"

    configs = find_kb_configs(vectordbs_dir)

  if not configs:
    logger.warning(f"No knowledgebase configurations found in {vectordbs_dir}")
    return f"No knowledgebase configurations found in {vectordbs_dir}"

  # Build output
  output = [f"Found {len(configs)} knowledgebase configuration(s)"]

  # Show system memory info
  memory_gb = args.memory_gb if hasattr(args, 'memory_gb') and args.memory_gb else get_system_memory_gb()
  settings_info = get_optimized_settings(memory_gb)
  gpu_info = settings_info.get('gpu_info', {})

  output.extend([
    f"\nSystem Memory: {memory_gb:.1f} GB",
    f"Optimization Tier: {settings_info['tier'].upper()}",
    f"  - Memory cache size: {settings_info['optimizations']['LIMITS']['memory_cache_size']}",
    f"  - Reference batch size: {settings_info['optimizations']['PERFORMANCE']['reference_batch_size']}",
    f"  - Thread pools: {settings_info['optimizations']['PERFORMANCE']['io_thread_pool_size']}"
  ])

  # Show GPU info if available
  if gpu_info and gpu_info.get('detected'):
    output.extend([
      f"\nGPU Detected: {gpu_info['gpu_memory_gb']:.1f} GB",
      f"  - Safe FAISS limit: {gpu_info['safe_limit_mb']} MB",
      f"  - GPU batch size: {settings_info['optimizations']['ALGORITHMS']['faiss_gpu_batch_size']}",
      f"  - Float16 mode: {settings_info['optimizations']['ALGORITHMS']['faiss_gpu_use_float16']}"
    ])

  if hasattr(args, 'analyze') and args.analyze:
    # Analyze mode - show KB sizes and recommendations
    total_size = 0
    kb_stats = []

    for config in configs:
      stats = analyze_kb_size(config)
      kb_stats.append(stats)
      total_size += stats['total_size_mb']

    # Sort by size
    kb_stats.sort(key=lambda x: x['total_size_mb'], reverse=True)

    output.extend([
      "\nKnowledgebase Analysis:",
      "-" * 80,
      f"{'KB Name':<30} {'DB (MB)':<10} {'Vector (MB)':<12} {'Total (MB)':<10}",
      "-" * 80
    ])

    for stats in kb_stats:
      output.append(
        f"{stats['kb_name']:<30} "
        f"{stats['db_size_mb']:<10.1f} "
        f"{stats['vector_size_mb']:<12.1f} "
        f"{stats['total_size_mb']:<10.1f}"
      )

    output.extend([
      "-" * 80,
      f"{'Total:':<30} {'':<10} {'':<12} {total_size:<10.1f}",
      "\nOptimization Recommendations:",
      "- Large KBs (>1GB) will benefit most from increased batch sizes",
      "- KBs with many queries benefit from larger memory caches"
    ])

    # Tier-specific recommendations
    tier_value = settings_info['tier']
    try:
      tier_enum = OptimizationTier.from_string(tier_value)
    except ValueError:
      tier_enum = None

    match tier_enum:
      case OptimizationTier.LOW:
        output.extend([
          "\nLow Memory Tier (<16GB):",
          "- Conservative settings to avoid memory pressure",
          "- Consider upgrading RAM for better performance"
        ])
      case OptimizationTier.MEDIUM:
        output.extend([
          "\nMedium Memory Tier (16-64GB):",
          "- Balanced settings for good performance",
          "- Suitable for most workloads"
        ])
      case OptimizationTier.HIGH:
        output.extend([
          "\nHigh Memory Tier (64-128GB):",
          "- High performance settings",
          "- Excellent for production workloads"
        ])
      case OptimizationTier.VERY_HIGH | None:
        output.extend([
          "\nVery High Memory Tier (>128GB):",
        "- Maximum performance settings",
        "- Optimal for large-scale deployments"
      ])
  else:
    # Optimization mode
    dry_run = hasattr(args, 'dry_run') and args.dry_run
    if dry_run:
      output.append("\nDRY RUN MODE - No changes will be made")

    all_changes = {}

    for config in configs:
      changes = optimize_config(config, dry_run=dry_run, memory_gb=memory_gb)
      if changes:
        all_changes[config] = changes
      output.append(format_changes(config, changes))

    if all_changes:
      output.append(
        f"\n{'Would optimize' if dry_run else 'Optimized'} "
        f"{len(all_changes)} configuration(s)"
      )

      if dry_run:
        output.append("\nTo apply these optimizations, run without --dry-run")
    else:
      output.append("\nAll configurations are already optimized!")

  return '\n'.join(output)


#fin
