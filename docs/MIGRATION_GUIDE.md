# CustomKB Migration Guide: New Knowledge Base Resolution System

## Overview

CustomKB has changed how knowledge bases are resolved and accessed. This guide helps you migrate from the old flexible path system to the new standardized approach.

## What Changed?

### Old System (Before)
- Accepted full paths: `/path/to/config.cfg`
- Accepted relative paths: `configs/myproject.cfg`
- Allowed config files anywhere on the filesystem
- Searched current directory first, then VECTORDBS

### New System (Now)
- Only accepts knowledge base names
- All KBs must be in VECTORDBS directory
- Automatically strips paths and `.cfg` extensions
- Configuration must be at: `VECTORDBS/<kb_name>/<kb_name>.cfg`

## Migration Steps

### Step 1: Identify Your Knowledge Bases

List all your existing knowledge base configuration files:
```bash
# Find all .cfg files
find / -name "*.cfg" -type f 2>/dev/null | grep -E "(customkb|knowledgebase|vectordb)"
```

### Step 2: Create VECTORDBS Directory Structure

```bash
# Set VECTORDBS location (default)
export VECTORDBS="/var/lib/vectordbs"

# Create the base directory
sudo mkdir -p $VECTORDBS
sudo chown $USER:$USER $VECTORDBS
```

### Step 3: Move Knowledge Bases

For each knowledge base, create a subdirectory and move all files:

```bash
# Example: Moving a KB from /home/user/projects/docs/
KB_NAME="myproject"
OLD_PATH="/home/user/projects/docs"

# Create KB directory
mkdir -p $VECTORDBS/$KB_NAME

# Move all KB files
mv $OLD_PATH/$KB_NAME.cfg $VECTORDBS/$KB_NAME/
mv $OLD_PATH/$KB_NAME.db $VECTORDBS/$KB_NAME/
mv $OLD_PATH/$KB_NAME.faiss $VECTORDBS/$KB_NAME/
mv $OLD_PATH/$KB_NAME.bm25 $VECTORDBS/$KB_NAME/ 2>/dev/null || true
mv $OLD_PATH/logs $VECTORDBS/$KB_NAME/ 2>/dev/null || true
```

### Step 4: Update Your Scripts

#### Command Line Usage

**Before:**
```bash
customkb query /home/user/projects/docs/myproject.cfg "search"
customkb query ../configs/myproject.cfg "search"
customkb query ./myproject.cfg "search"
```

**After:**
```bash
customkb query myproject "search"
```

#### Shell Scripts

**Before:**
```bash
#!/bin/bash
CONFIG="/path/to/myproject.cfg"
customkb database "$CONFIG" docs/*.txt
customkb embed "$CONFIG"
customkb query "$CONFIG" "test query"
```

**After:**
```bash
#!/bin/bash
KB_NAME="myproject"
customkb database "$KB_NAME" docs/*.txt
customkb embed "$KB_NAME"
customkb query "$KB_NAME" "test query"
```

#### Python Integration

**Before:**
```python
import subprocess

config_path = "/path/to/myproject.cfg"
result = subprocess.run(
    ["customkb", "query", config_path, "search term"],
    capture_output=True
)
```

**After:**
```python
import subprocess

kb_name = "myproject"
result = subprocess.run(
    ["customkb", "query", kb_name, "search term"],
    capture_output=True
)
```

### Step 5: Verify Migration

Check that all KBs are accessible:
```bash
# List all KBs in VECTORDBS
ls -la $VECTORDBS/

# Test each KB
for kb in $VECTORDBS/*/; do
    kb_name=$(basename "$kb")
    echo "Testing $kb_name..."
    customkb verify-indexes "$kb_name"
done
```

## Common Migration Scenarios

### Scenario 1: Multiple Config Files, Same Name

If you have multiple `project.cfg` files in different locations:

```bash
# Rename KBs to be unique
mv /path1/project.cfg $VECTORDBS/project-docs/project-docs.cfg
mv /path2/project.cfg $VECTORDBS/project-api/project-api.cfg
```

### Scenario 2: Domain-Style Names

Domain-style names work without changes:
```bash
# Before: /anywhere/example.com.cfg
# After: $VECTORDBS/example.com/example.com.cfg
```

### Scenario 3: Shared Knowledge Bases

For teams sharing KBs:
```bash
# Set consistent VECTORDBS for all users
echo 'export VECTORDBS="/shared/vectordbs"' >> /etc/profile.d/customkb.sh
```

## Troubleshooting

### Error: "Knowledge base 'name' not found"

**Cause:** KB doesn't exist in VECTORDBS directory

**Solution:**
1. Check VECTORDBS is set: `echo $VECTORDBS`
2. List available KBs: `ls $VECTORDBS/`
3. Create missing KB directory and move files

### Error: "Permission denied"

**Cause:** Incorrect permissions on VECTORDBS

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER:$USER $VECTORDBS
chmod -R 755 $VECTORDBS
```

### Old Scripts Still Using Paths

**Quick Fix:** Create wrapper script
```bash
#!/bin/bash
# customkb-compat.sh - Compatibility wrapper

# Extract KB name from path
KB_INPUT="$2"
KB_NAME=$(basename "$KB_INPUT" .cfg)

# Replace second argument with KB name
set -- "$1" "$KB_NAME" "${@:3}"

# Call real customkb
exec customkb "$@"
```

## Benefits of the New System

1. **Simpler Commands**: Just use KB names, no paths needed
2. **Better Organization**: All KBs in one standard location
3. **Improved Security**: No path traversal concerns
4. **Helpful Errors**: Shows available KBs when one isn't found
5. **Consistency**: Same KB resolution across all commands

## Quick Reference

### Before → After Examples

| Old Command | New Command |
|------------|-------------|
| `customkb query /path/to/project.cfg "search"` | `customkb query project "search"` |
| `customkb embed ../configs/api.cfg` | `customkb embed api` |
| `customkb database ./docs.cfg files/*.txt` | `customkb database docs files/*.txt` |
| `customkb edit /var/lib/old/kb.cfg` | `customkb edit kb` |

### New Directory Structure

```
$VECTORDBS/
├── myproject/
│   ├── myproject.cfg      # Required: Config file
│   ├── myproject.db       # Required: SQLite database
│   ├── myproject.faiss    # Required: Vector index
│   ├── myproject.bm25     # Optional: BM25 index
│   └── logs/              # Optional: Log directory
├── api-docs/
│   ├── api-docs.cfg
│   ├── api-docs.db
│   └── api-docs.faiss
```

## Getting Help

If you encounter issues during migration:

1. Check the error message - it often lists available KBs
2. Run with debug mode: `customkb query myproject "test" --debug`
3. Check logs: `$VECTORDBS/<kb_name>/logs/<kb_name>.log`
4. File an issue: https://github.com/Open-Technology-Foundation/customkb/issues

---

*This migration is a one-time change that simplifies CustomKB usage going forward.*

#fin