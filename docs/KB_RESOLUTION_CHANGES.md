# Knowledge Base Resolution Changes

## Overview

The knowledge base parameter resolution system has been changed to require that all knowledge bases exist as subdirectories within the VECTORDBS directory (`/var/lib/vectordbs` by default).

## Breaking Changes

### Old Behavior
- Accepted full paths: `/path/to/config.cfg`
- Accepted relative paths: `configs/myproject.cfg`
- Searched current directory first, then VECTORDBS
- Allowed configuration files anywhere on the filesystem

### New Behavior
- Only accepts knowledge base names
- Automatically strips paths and `.cfg` extensions
- Requires knowledge base to exist as a subdirectory in VECTORDBS
- Configuration file must be at: `VECTORDBS/<kb_name>/<kb_name>.cfg`

## Examples

### Input Resolution
```bash
# All of these now resolve to the same KB:
customkb query okusimail "test"
customkb query okusimail.cfg "test"
customkb query /path/to/okusimail "test"
customkb query /path/to/okusimail.cfg "test"

# Result: Uses /var/lib/vectordbs/okusimail/okusimail.cfg
```

### Error Messages
```bash
# Old error:
Error: Configuration file not found.

# New error:
Error: Knowledge base 'myproject' not found in /var/lib/vectordbs
Available knowledge bases: okusimail, okusiassociates, jakartapost
```

## Migration Guide

### For Users
1. Move all knowledge bases to VECTORDBS directory
2. Ensure each KB has its own subdirectory
3. Update scripts to use KB names instead of paths

### Directory Structure
```
/var/lib/vectordbs/
├── okusimail/
│   ├── okusimail.cfg
│   ├── okusimail.db
│   ├── okusimail.faiss
│   └── logs/
├── okusiassociates/
│   ├── okusiassociates.cfg
│   ├── okusiassociates.db
│   └── okusiassociates.faiss
```

## Benefits

1. **Simpler Usage**: Just specify the KB name
2. **Improved Security**: No path traversal concerns
3. **Consistency**: All KBs in one standard location
4. **Better Error Messages**: Shows available KBs when one isn't found
5. **Cleaner Commands**: No need to remember paths or extensions

## Implementation Details

### New Functions

#### `get_kb_name(kb_input: str) -> Optional[str]`
- Extracts clean KB name from user input
- Strips paths and `.cfg` extensions
- Validates KB exists in VECTORDBS
- Lists available KBs on error

#### Updated `get_fq_cfg_filename(cfgfile: str) -> Optional[str]`
- Now just calls `get_kb_name()` and constructs the path
- Much simpler implementation
- Always returns `VECTORDBS/<kb_name>/<kb_name>.cfg`

### Files Modified
- `config/config_manager.py`: Core resolution logic
- `customkb.py`: Updated help text and error messages
- `database/db_manager.py`: Updated error messages
- `embedding/embed_manager.py`: Updated error messages
- `query/query_manager.py`: Updated error messages

## Testing

Test script to verify the new behavior:
```python
from config.config_manager import get_kb_name, get_fq_cfg_filename

test_cases = [
    'okusimail',                    # Simple name
    'okusimail.cfg',                # With extension
    '/path/to/okusimail',           # With path
    '/path/to/okusimail.cfg',       # Full path
    'nonexistent',                  # Should fail
]

for test in test_cases:
    kb_name = get_kb_name(test)
    if kb_name:
        config_path = get_fq_cfg_filename(test)
        print(f"{test} → {config_path}")
    else:
        print(f"{test} → Not found")
```