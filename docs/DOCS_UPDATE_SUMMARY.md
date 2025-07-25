# Documentation Updates Summary

## Files Updated for New KB Resolution System

### 1. CLAUDE.md
- Replaced "Knowledge Base Configuration Paths" section with "Knowledge Base Resolution System"
- Added clear examples showing how all input formats resolve to KB names
- Updated error handling documentation
- Added migration guide section

### 2. README.md (11 changes)
- Updated Quick Start section to create KB directory first
- Changed all command examples from `.cfg` files to KB names
- Replaced flexible path handling with standardized KB structure
- Updated Python integration example
- Fixed troubleshooting section for new error messages
- Updated Basic Usage Example steps

### 3. docs/performance_optimization_guide.md (4 changes)
- Changed `./optimize-kb` references to `customkb optimize`
- Updated all example commands to use KB names
- Fixed command syntax for new resolution system

### 4. docs/GPU_ACCELERATION.md (2 changes)
- Changed `knowledge_base_name` to `kb_name` for consistency
- Updated `kb_config` parameter to `kb_name`

### 5. New Documentation Created
- `KB_RESOLUTION_CHANGES.md` - Breaking changes documentation
- `MIGRATION_GUIDE.md` - Comprehensive migration guide
- `KB_RESOLUTION_SUMMARY.md` - Implementation summary
- `DOCS_UPDATE_SUMMARY.md` - This file

## Key Documentation Changes

### Before
```bash
# Various ways to specify KBs
customkb query /path/to/config.cfg "search"
customkb query ../project/config.cfg "search"
customkb query ./myproject.cfg "search"
```

### After
```bash
# Only KB names
customkb query myproject "search"
```

### Error Messages
**Old**: "Configuration file not found"
**New**: "Knowledge base 'name' not found in /var/lib/vectordbs"

### Directory Structure
All KBs must now be organized as:
```
$VECTORDBS/
├── kb_name/
│   ├── kb_name.cfg
│   ├── kb_name.db
│   ├── kb_name.faiss
│   └── logs/
```

## Documentation Still Accurate
- GPU_MEMORY_MANAGEMENT.md - Uses KB names correctly
- SAFE_TESTING_GUIDE.md - References to configuration are about file content, not paths

## Notes
- All command examples now use KB names instead of paths
- Documentation emphasizes the standardized structure
- Migration guides help users transition existing installations

#fin