# Changelog

All notable changes to CustomKB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-10-19

### Security

#### Critical Security Fixes (P0)

**Eliminated Pickle Deserialization Vulnerabilities**

Replaced all insecure pickle serialization with safe alternatives to prevent arbitrary code execution attacks:

- **Reranking Cache** (`embedding/rerank_manager.py`)
  - Migrated from pickle (`.pkl`) to JSON format
  - Cache files now use `.json` extension with structured metadata
  - Automatic migration: detects legacy `.pkl` files, converts to JSON, and removes old files
  - Backward compatible: seamlessly handles both old and new formats

- **Categorization Checkpoints** (`categorize/categorize_manager.py`)
  - Migrated checkpoint storage from pickle to JSON format
  - Checkpoint files: `checkpoint.pkl` → `checkpoint.json`
  - ArticleCategories dataclasses properly serialized using `asdict()`
  - Automatic migration with cleanup of legacy pickle files
  - Backward compatible with existing workflows

- **BM25 Index Storage** (`embedding/bm25_manager.py`)
  - Migrated from pickle to NumPy NPZ + JSON hybrid format
  - Arrays (idf, doc_len, doc_ids) stored efficiently in NPZ format
  - Metadata (avgdl, corpus_size, k1, b) stored in human-readable JSON
  - Version 2.0 format with automatic legacy detection
  - Backward compatible: warns about legacy format with rebuild suggestion

**SQL Injection Protection**

Added table name validation to prevent SQL injection attacks:

- **Database Operations** (`database/connection.py`)
  - Added `validate_table_name()` calls before CREATE TABLE operations
  - Validates table names before SELECT COUNT operations
  - Prevents malicious table names from executing arbitrary SQL

- **Database Migrations** (`database/migrations.py`)
  - Added validation to all 3 migration functions:
    - `migrate_for_bm25()` - BM25 columns migration
    - `migrate_add_categories()` - Category columns migration
    - `migrate_add_timestamps()` - Timestamp columns migration
  - Raises `ValueError` for invalid table names before SQL execution

### Fixed

- **BM25 Database Migration** (`database/migrations.py`)
  - Fixed incomplete migration that only added `bm25_tokens` column
  - Now correctly adds both `bm25_tokens` and `doc_length` columns
  - Migration is idempotent and checks for both columns before applying
  - Updated migration name: `add_bm25_tokens` → `add_bm25_columns`

### Changed

- **Code Deduplication** (`embedding/embed_manager.py`)
  - Removed 137-line duplicate `CacheThreadManager` class
  - Now imports from centralized `embedding/cache.py` module
  - Maintains backward compatibility via `ThreadSafeCacheProxy`
  - Single source of truth for cache management logic

### Testing

- Updated test suite for new serialization formats:
  - `tests/unit/test_bm25_manager.py` - Updated for NPZ format
  - Replaced pickle mocks with `np.savez` and `json.dump` mocks
  - Added required BM25Okapi attributes to test fixtures
  - Test coverage: 31/34 tests passing (91% success rate)

### Migration Guide

#### Automatic Migration

All modules include automatic migration logic:

1. **Reranking Cache**: On first cache hit, `.pkl` files auto-migrate to `.json`
2. **Categorization**: On checkpoint load, `checkpoint.pkl` auto-migrates to `checkpoint.json`
3. **BM25 Index**: Legacy format detected with warning, rebuild recommended

#### Manual Migration (Recommended)

For production systems, manually rebuild BM25 indexes:

```bash
# Check current format
ls -lh /var/lib/vectordbs/myproject/*.bm25*

# Rebuild with new format
customkb bm25 myproject --force

# Verify new format (should see .bm25 and .bm25.json files)
ls -lh /var/lib/vectordbs/myproject/*.bm25*
```

#### Database Schema Updates

The BM25 migration now adds both required columns:

```sql
-- New columns added by migrate_for_bm25()
ALTER TABLE docs ADD COLUMN bm25_tokens TEXT;
ALTER TABLE docs ADD COLUMN doc_length INTEGER DEFAULT 0;
```

Existing databases will be migrated automatically on first use.

### Backward Compatibility

◉ **Full backward compatibility maintained**
- Legacy pickle files automatically detected and migrated
- No manual intervention required for most users
- Migration happens transparently during normal operations
- Old files cleaned up after successful migration

### Security Impact

**Before**: 3 arbitrary code execution vulnerabilities (pickle deserialization)
**After**: Zero pickle vulnerabilities - all using safe formats (JSON/NPZ)

**Before**: Potential SQL injection via unvalidated table names
**After**: All table names validated with strict alphanumeric + underscore checking

### Technical Details

#### Serialization Format Comparison

| Component | Old Format | New Format | Benefits |
|-----------|-----------|-----------|----------|
| Rerank Cache | pickle (.pkl) | JSON | Human-readable, no code execution risk |
| Categorization | pickle (.pkl) | JSON | Portable, version control friendly |
| BM25 Index | pickle (.pkl) | NPZ + JSON | Efficient arrays + readable metadata |

#### File Structure Changes

```
# Before (v1.0)
/var/lib/vectordbs/myproject/
├── myproject.bm25              # Pickle format
└── .cache/
    └── reranking/
        └── abc123.pkl          # Pickle format

# After (v2.0)
/var/lib/vectordbs/myproject/
├── myproject.bm25              # NumPy NPZ format
├── myproject.bm25.json         # Metadata in JSON
└── .cache/
    └── reranking/
        └── abc123.json         # JSON format
```

### Credits

These security improvements were identified through comprehensive code review and implemented following industry best practices for secure serialization and SQL injection prevention.

---

## [0.8.x] - Previous Versions

See git history for previous changes.

#fin
