"""
Categorization manager for CustomKB.

This module handles AI-powered categorization of articles in knowledgebases.
"""

import asyncio
import json
import sqlite3
import sys
import os
import argparse
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
import signal
import pickle
import re

from openai import AsyncOpenAI, OpenAI
import httpx
from tqdm.asyncio import tqdm_asyncio
import pandas as pd

# Import from customkb modules
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from utils.logging_config import get_logger
from utils.text_utils import get_env
from utils.security_utils import validate_file_path
from categorize.category_deduplicator import CategoryDeduplicator

# Setup module logger
logger = get_logger(__name__)

@dataclass
class CategoryResult:
  """Represents a category assignment with confidence"""
  name: str
  confidence: float
  
@dataclass
class ArticleCategories:
  """Complete categorization result for an article"""
  article_path: str
  total_chunks: int
  sampled_chunks: int
  categories: List[CategoryResult]
  primary_category: Optional[str]
  processing_time: float
  error: Optional[str] = None
  model_used: Optional[str] = None
  suggested_new_categories: List[str] = field(default_factory=list)

class CategoryGenerator:
  """Generate initial categories based on KB content"""
  
  def __init__(self, kb: KnowledgeBase):
    self.kb = kb
    self.logger = get_logger(__name__)
    
  def analyze_content(self, sample_size: int = 10) -> List[str]:
    """Analyze KB content to generate initial categories"""
    # Use existing database connection from KnowledgeBase
    conn = sqlite3.connect(self.kb.knowledge_base_db)
    cursor = conn.cursor()
    
    # Check table structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'docs')")
    table = cursor.fetchone()
    if not table:
      raise ValueError(f"No 'chunks' or 'docs' table found in {self.kb.knowledge_base_db}")
    
    table_name = table[0]
    
    # Validate table name to prevent SQL injection
    if table_name not in ['docs', 'chunks']:
      logger.error(f"Invalid table name: {table_name}")
      return {}
    
    # Get sample articles - using validated table name
    if table_name == 'docs':
      cursor.execute("""
        SELECT DISTINCT sourcedoc 
        FROM docs
        ORDER BY RANDOM()
        LIMIT ?
      """, (sample_size,))
    else:
      cursor.execute("""
        SELECT DISTINCT sourcedoc 
        FROM chunks
        ORDER BY RANDOM()
        LIMIT ?
      """, (sample_size,))
    
    sample_docs = cursor.fetchall()
    
    # Analyze content patterns
    all_text = []
    for doc in sample_docs:
      if table_name == 'docs':
        cursor.execute("""
          SELECT embedtext 
          FROM docs
          WHERE sourcedoc = ?
          LIMIT 5
        """, (doc[0],))
      else:
        cursor.execute("""
          SELECT embedtext 
          FROM chunks
          WHERE sourcedoc = ?
          LIMIT 5
        """, (doc[0],))
      
      chunks = cursor.fetchall()
      doc_text = ' '.join([chunk[0] for chunk in chunks if chunk[0]])
      all_text.append(doc_text[:1000])  # First 1000 chars
    
    conn.close()
    
    # Generate categories using AI
    categories = self._generate_categories_from_samples(all_text)
    return categories
  
  def _generate_categories_from_samples(self, samples: List[str]) -> List[str]:
    """Use AI to generate categories from sample texts"""
    client = OpenAI(api_key=get_env('OPENAI_API_KEY', None))
    
    combined_text = '\n\n---\n\n'.join(samples[:5])  # Use first 5 samples
    
    prompt = f"""Analyze these document samples and suggest 15-20 broad topic categories that would cover a knowledgebase containing such documents.

Document Samples:
{combined_text[:3000]}

Return ONLY a Python list of category names, like:
["Technology", "Business", "Healthcare", ...]

Categories should be:
- Broad enough to group related articles
- Specific enough to be meaningful
- Single or two-word phrases
- Diverse to cover different topics

Return ONLY the Python list, no other text."""

    try:
      response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
      )
      
      categories_text = response.choices[0].message.content.strip()
      # Parse the list safely using json.loads instead of eval
      categories = json.loads(categories_text)
      
      if isinstance(categories, list):
        return categories
      else:
        raise ValueError("AI did not return a list")
        
    except Exception as e:
      self.logger.error(f"Failed to generate categories: {e}")
      # Fallback categories
      return [
        "Technology", "Business", "Science", "Healthcare", "Education",
        "Finance", "Legal", "Marketing", "Engineering", "Research",
        "Culture", "Politics", "Environment", "Social Issues", "History"
      ]

class AdaptiveCategorizer:
  """Main categorization engine with adaptive learning"""
  
  def __init__(self, kb: KnowledgeBase, model_name: Optional[str] = None,
               checkpoint_file: Optional[str] = None,
               sampling_config: Optional[Tuple[int, int, int]] = None,
               confidence_threshold: Optional[float] = None,
               enable_deduplication: bool = True,
               dedup_threshold: float = 85.0,
               use_variable_categories: bool = True):
    self.kb = kb
    self.model_name = model_name or "gpt-4o-mini"
    self.checkpoint_file = checkpoint_file
    self.sampling_config = sampling_config or (3, 3, 3)
    self.confidence_threshold = confidence_threshold or 0.5
    self.enable_deduplication = enable_deduplication
    self.dedup_threshold = dedup_threshold
    self.use_variable_categories = use_variable_categories
    self.logger = get_logger(__name__)
    
    # Load or generate categories
    self.categories = self._load_or_generate_categories()
    self.dynamic_categories = set()
    self.processed_articles = []
    self.checkpoint_counter = 0
    
    # Setup OpenAI client
    api_key = get_env('OPENAI_API_KEY', None)
    if not api_key:
      raise ValueError("OPENAI_API_KEY environment variable not set")
    
    self.client = AsyncOpenAI(
      api_key=api_key,
      timeout=httpx.Timeout(60.0, connect=10.0),
      max_retries=3
    )
    
    # Setup deduplicator if enabled
    self.deduplicator = CategoryDeduplicator() if enable_deduplication else None
    
  def _load_or_generate_categories(self) -> List[str]:
    """Load existing categories or generate new ones"""
    cats_dir = Path(self.kb.knowledge_base_db).parent / "cats"
    categories_file = cats_dir / "categories.yaml"
    
    if categories_file.exists():
      with open(categories_file, 'r') as f:
        data = yaml.safe_load(f)
        return data.get('categories', [])
    else:
      # Generate categories from content
      generator = CategoryGenerator(self.kb)
      categories = generator.analyze_content()
      
      # Save categories
      cats_dir.mkdir(exist_ok=True)
      with open(categories_file, 'w') as f:
        yaml.dump({
          'categories': categories,
          'generated_at': datetime.now().isoformat(),
          'kb_name': self.kb.knowledge_base_name
        }, f)
      
      self.logger.info(f"Generated {len(categories)} initial categories")
      return categories
  
  def _calculate_complexity(self, text: str) -> int:
    """Calculate article complexity to determine number of categories"""
    if not self.use_variable_categories:
      return 5  # Default to 5 categories
    
    # Simple heuristics for complexity
    word_count = len(text.split())
    unique_words = len(set(text.lower().split()))
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    # Complexity score
    complexity = 0
    
    # Length factor
    if word_count > 2000:
      complexity += 3
    elif word_count > 1000:
      complexity += 2
    elif word_count > 500:
      complexity += 1
    
    # Vocabulary diversity
    if unique_words > 500:
      complexity += 2
    elif unique_words > 250:
      complexity += 1
    
    # Sentence complexity
    avg_sentence_length = word_count / max(sentence_count, 1)
    if avg_sentence_length > 25:
      complexity += 1
    
    # Map to category count (3-7 categories)
    if complexity >= 5:
      return 7
    elif complexity >= 3:
      return 6
    elif complexity >= 2:
      return 5
    elif complexity >= 1:
      return 4
    else:
      return 3
  
  def _sample_chunks(self, chunks: List[Tuple], top: int, middle: int, bottom: int) -> str:
    """Sample chunks from article"""
    total = len(chunks)
    sampled = []
    
    # Top chunks
    sampled.extend(chunks[:min(top, total)])
    
    # Middle chunks
    if middle > 0 and total > top + bottom:
      mid_start = total // 2 - middle // 2
      mid_end = mid_start + middle
      sampled.extend(chunks[mid_start:mid_end])
    
    # Bottom chunks
    if bottom > 0 and total > top:
      sampled.extend(chunks[-min(bottom, total - top):])
    
    # Combine text
    text = '\n'.join([chunk[1] for chunk in sampled if chunk[1]])
    return text
  
  async def categorize_article(self, article_path: str, chunks: List[Tuple]) -> ArticleCategories:
    """Categorize a single article"""
    start_time = time.time()
    
    # Sample chunks
    top, middle, bottom = self.sampling_config
    sampled_text = self._sample_chunks(chunks, top, middle, bottom)
    
    # Calculate complexity
    num_categories = self._calculate_complexity(sampled_text)
    
    # Build prompt
    all_categories = list(self.categories) + list(self.dynamic_categories)
    categories_str = ', '.join(all_categories)
    
    prompt = f"""Categorize this article into the {num_categories} most relevant categories.

Available Categories: {categories_str}

Article Text:
{sampled_text[:4000]}

Instructions:
1. Select exactly {num_categories} categories that best describe this article
2. Assign confidence scores (0.0-1.0) for each category
3. Order by relevance (most relevant first)
4. If no existing category fits well (confidence < {self.confidence_threshold}), suggest a new category name

Return ONLY a JSON object like:
{{
  "categories": [
    {{"name": "Category1", "confidence": 0.95}},
    {{"name": "Category2", "confidence": 0.85}}
  ],
  "suggested_new": ["NewCategory1"]
}}"""

    try:
      response = await self.client.chat.completions.create(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
        response_format={"type": "json_object"}
      )
      
      # Get response content
      response_content = response.choices[0].message.content
      
      # Try to parse JSON with better error handling
      try:
        result = json.loads(response_content)
      except json.JSONDecodeError as json_err:
        # Log the actual response for debugging
        self.logger.debug(f"Invalid JSON response: {response_content[:500]}")
        
        # Try to clean up common JSON issues
        cleaned_content = response_content.strip()
        
        # Remove any trailing commas before closing brackets/braces
        cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
        
        # Try parsing again with cleaned content
        try:
          result = json.loads(cleaned_content)
        except json.JSONDecodeError:
          # If still fails, create a fallback response
          self.logger.warning(f"Could not parse JSON for {article_path}, using fallback")
          result = {
            "categories": [],
            "suggested_new": []
          }
      
      # Parse categories with validation
      categories = []
      for cat in result.get('categories', []):
        if isinstance(cat, dict) and 'name' in cat and 'confidence' in cat:
          try:
            categories.append(CategoryResult(
              name=str(cat['name']),
              confidence=float(cat['confidence'])
            ))
          except (ValueError, TypeError) as e:
            self.logger.debug(f"Invalid category data: {cat}, error: {e}")
            continue
      
      # Add suggested categories to dynamic set
      suggested = result.get('suggested_new', [])
      if suggested and self.confidence_threshold:
        for cat in categories:
          if cat.confidence < self.confidence_threshold:
            self.dynamic_categories.update(suggested)
            break
      
      return ArticleCategories(
        article_path=article_path,
        total_chunks=len(chunks),
        sampled_chunks=len(sampled_text.split('\n')),
        categories=categories,
        primary_category=categories[0].name if categories else None,
        processing_time=time.time() - start_time,
        model_used=self.model_name,
        suggested_new_categories=suggested
      )
      
    except Exception as e:
      self.logger.error(f"Error categorizing {article_path}: {e}")
      return ArticleCategories(
        article_path=article_path,
        total_chunks=len(chunks),
        sampled_chunks=0,
        categories=[],
        primary_category=None,
        processing_time=time.time() - start_time,
        error=str(e)
      )
  
  async def process_all_articles(self, articles: List[str], max_concurrent: int = 5) -> List[ArticleCategories]:
    """Process all articles with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(article_data):
      async with semaphore:
        return await self.categorize_article(article_data[0], article_data[1])
    
    # Get chunks for each article
    conn = sqlite3.connect(self.kb.knowledge_base_db)
    cursor = conn.cursor()
    
    # Determine table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'docs')")
    table_name = cursor.fetchone()[0]
    
    # Validate table name to prevent SQL injection
    if table_name not in ['docs', 'chunks']:
      logger.error(f"Invalid table name: {table_name}")
      conn.close()
      return []
    
    article_data = []
    for article in articles:
      if table_name == 'docs':
        cursor.execute("""
          SELECT sid, embedtext 
          FROM docs
          WHERE sourcedoc = ?
          ORDER BY sid
        """, (article,))
      else:
        cursor.execute("""
          SELECT sid, embedtext 
          FROM chunks
          WHERE sourcedoc = ?
          ORDER BY sid
        """, (article,))
      chunks = cursor.fetchall()
      article_data.append((article, chunks))
    
    conn.close()
    
    # Process with progress bar
    tasks = [process_with_limit(data) for data in article_data]
    
    results = []
    with tqdm_asyncio(total=len(tasks), desc="Categorizing articles") as pbar:
      for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        pbar.update(1)
        
        # Checkpoint every 10 articles
        self.checkpoint_counter += 1
        if self.checkpoint_counter % 10 == 0:
          self._save_checkpoint(results)
    
    return results
  
  def _save_checkpoint(self, results: List[ArticleCategories]):
    """Save checkpoint for resume capability"""
    if not self.checkpoint_file:
      cats_dir = Path(self.kb.knowledge_base_db).parent / "cats"
      self.checkpoint_file = cats_dir / "checkpoint.pkl"
    
    checkpoint_data = {
      'results': results,
      'dynamic_categories': list(self.dynamic_categories),
      'timestamp': datetime.now().isoformat()
    }
    
    with open(self.checkpoint_file, 'wb') as f:
      pickle.dump(checkpoint_data, f)
    
    self.logger.debug(f"Checkpoint saved: {len(results)} articles processed")
  
  def load_checkpoint(self) -> Optional[List[ArticleCategories]]:
    """Load checkpoint if exists"""
    if self.checkpoint_file and Path(self.checkpoint_file).exists():
      with open(self.checkpoint_file, 'rb') as f:
        data = pickle.load(f)
        self.dynamic_categories = set(data.get('dynamic_categories', []))
        return data.get('results', [])
    return None

def process_categorize(args: argparse.Namespace, logger) -> str:
  """
  Main entry point for categorization command.
  
  Args:
      args: Command-line arguments
      logger: Logger instance
      
  Returns:
      Status message
  """
  # Get KB configuration
  config_file = get_fq_cfg_filename(args.config_file)
  if not config_file:
    return f"Error: Knowledgebase '{args.config_file}' not found"
  
  kb = KnowledgeBase(config_file)
  logger.info(f"Processing categorization for: {kb.knowledge_base_name}")
  
  # Check if listing categories
  if hasattr(args, 'list_categories') and args.list_categories:
    return list_categories(kb, logger)
  
  # Run categorization
  return asyncio.run(categorize_async(args, kb, logger))

async def categorize_async(args: argparse.Namespace, kb: KnowledgeBase, logger) -> str:
  """
  Async categorization process.
  """
  # Get articles to process
  conn = sqlite3.connect(kb.knowledge_base_db)
  cursor = conn.cursor()
  
  # Check table
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'docs')")
  table = cursor.fetchone()
  if not table:
    return f"Error: No 'chunks' or 'docs' table found in database"
  
  table_name = table[0]
  
  # Validate table name to prevent SQL injection
  if table_name not in ['docs', 'chunks']:
    conn.close()
    return f"Error: Invalid table name: {table_name}"
  
  # Get all unique articles - using validated table name
  if table_name == 'docs':
    cursor.execute("SELECT DISTINCT sourcedoc FROM docs")
  else:
    cursor.execute("SELECT DISTINCT sourcedoc FROM chunks")
  all_articles = [row[0] for row in cursor.fetchall()]
  conn.close()
  
  logger.info(f"Found {len(all_articles)} articles in knowledgebase")
  
  # Determine sample size
  if hasattr(args, 'sample') and args.sample:
    articles = all_articles[:args.sample]
    logger.info(f"Processing sample of {len(articles)} articles")
  else:
    articles = all_articles
    logger.info(f"Processing all {len(articles)} articles")
  
  # Setup categorizer
  sampling_config = (3, 3, 3)  # Default sampling
  if hasattr(args, 'sampling') and args.sampling:
    parts = args.sampling.split('-')
    if len(parts) == 3:
      sampling_config = tuple(map(int, parts))
  
  categorizer = AdaptiveCategorizer(
    kb=kb,
    model_name=getattr(args, 'model', 'gpt-4o-mini'),
    sampling_config=sampling_config,
    confidence_threshold=getattr(args, 'confidence_threshold', 0.5),
    enable_deduplication=getattr(args, 'enable_deduplication', True),
    dedup_threshold=getattr(args, 'dedup_threshold', 85.0),
    use_variable_categories=getattr(args, 'use_variable_categories', True)
  )
  
  # Check for checkpoint
  existing_results = []
  if hasattr(args, 'resume') and args.resume:
    existing_results = categorizer.load_checkpoint() or []
    if existing_results:
      processed_articles = {r.article_path for r in existing_results}
      articles = [a for a in articles if a not in processed_articles]
      logger.info(f"Resuming from checkpoint: {len(existing_results)} already processed")
  
  # Process articles
  if articles:
    results = await categorizer.process_all_articles(
      articles,
      max_concurrent=getattr(args, 'max_concurrent', 5)
    )
    results = existing_results + results
  else:
    results = existing_results
  
  # Deduplicate categories if enabled
  if categorizer.enable_deduplication and categorizer.deduplicator:
    all_categories = set()
    for result in results:
      all_categories.update([cat.name for cat in result.categories])
    
    duplicates = categorizer.deduplicator.find_duplicates(list(all_categories))
    if duplicates:
      logger.info(f"Found {len(duplicates)} groups of similar categories")
      # Apply deduplication to each result's categories
      for result in results:
        category_names = [cat.name for cat in result.categories]
        deduplicated_names = categorizer.deduplicator.apply_to_results(category_names)
        # Update categories with deduplicated names
        result.categories = [CategoryResult(name=name, confidence=next((cat.confidence for cat in result.categories if cat.name == name or name in deduplicated_names), 0.0)) for name in deduplicated_names]
        # Update primary category if it was changed
        if result.primary_category:
          deduplicated_primary = categorizer.deduplicator.apply_to_results([result.primary_category])
          result.primary_category = deduplicated_primary[0] if deduplicated_primary else result.primary_category
  
  # Save results
  output_dir = Path(kb.knowledge_base_db).parent / "cats"
  output_dir.mkdir(exist_ok=True)
  
  # Save JSON
  output_file = output_dir / "categorization.json"
  with open(output_file, 'w') as f:
    json.dump([asdict(r) for r in results], f, indent=2)
  
  # Save CSV
  csv_file = output_dir / "categorization.csv"
  df = pd.DataFrame([
    {
      'article': r.article_path,
      'primary_category': r.primary_category,
      'all_categories': ', '.join([c.name for c in r.categories]),
      'confidence': r.categories[0].confidence if r.categories else 0,
      'processing_time': r.processing_time,
      'error': r.error
    }
    for r in results
  ])
  df.to_csv(csv_file, index=False)
  
  # Generate summary
  successful = len([r for r in results if not r.error])
  failed = len([r for r in results if r.error])
  
  all_categories = set()
  for r in results:
    all_categories.update([c.name for c in r.categories])
  
  summary = f"""
Categorization Complete
======================
Articles processed: {len(results)}
Successful: {successful}
Failed: {failed}
Unique categories: {len(all_categories)}
Dynamic categories added: {len(categorizer.dynamic_categories)}

Results saved to:
- {output_file}
- {csv_file}
"""
  
  # Save summary
  summary_file = output_dir / "summary.txt"
  with open(summary_file, 'w') as f:
    f.write(summary)
  
  # Import to database if requested
  if hasattr(args, 'import_to_db') and args.import_to_db:
    from categorize.import_to_db import import_categories
    import_result = import_categories(kb, results)
    summary += f"\n{import_result}"
  
  logger.info(summary)
  return summary

def list_categories(kb: KnowledgeBase, logger) -> str:
  """List existing categories for a knowledgebase"""
  cats_file = Path(kb.knowledge_base_db).parent / "cats" / "categorization.json"
  
  if not cats_file.exists():
    return "No categorization results found. Run 'customkb categorize' first."
  
  with open(cats_file, 'r') as f:
    results = json.load(f)
  
  # Count categories
  category_counts = {}
  for article in results:
    for cat in article.get('categories', []):
      name = cat['name']
      category_counts[name] = category_counts.get(name, 0) + 1
  
  # Sort by count
  sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
  
  output = f"Categories in {kb.knowledge_base_name}:\n"
  output += "=" * 50 + "\n"
  for cat, count in sorted_cats:
    output += f"{cat:30} {count:5} articles\n"
  
  output += f"\nTotal: {len(sorted_cats)} unique categories"
  
  return output

#fin