# CustomKB Logging Standards

## Overview

This document defines the logging standards for the CustomKB project. All modules should follow these guidelines to ensure consistent, useful, and performant logging across the codebase.

## Quick Start

### Basic Setup

```python
from utils.logging_config import get_logger

# At module level
logger = get_logger(__name__)

# Use in functions
def process_data(data):
    logger.debug(f"Processing {len(data)} items")
    try:
        result = do_processing(data)
        logger.info(f"Successfully processed {len(result)} items")
        return result
    except Exception as e:
        logger.error(f"Failed to process data: {e}", exc_info=True)
        raise
```

### Migration from Old Pattern

**Old:**
```python
from utils.logging_utils import get_logger
logger = get_logger(__name__)
```

**New:**
```python
from utils.logging_config import get_logger
logger = get_logger(__name__)
```

## Log Levels

Use the appropriate log level for each message:

| Level | Use Case | Example |
|-------|----------|---------|
| DEBUG | Detailed diagnostic info | `logger.debug(f"Cache hit for key: {key}")` |
| INFO | Normal program flow | `logger.info(f"Processing batch {i}/{total}")` |
| WARNING | Recoverable issues | `logger.warning(f"Retrying after error: {e}")` |
| ERROR | Errors that need attention | `logger.error(f"Failed to connect: {e}")` |
| CRITICAL | System failures | `logger.critical("Database corrupted")` |

## Best Practices

### 1. Module-Level Logger

Always create logger at module level:

```python
# Good
logger = get_logger(__name__)

class MyClass:
    def method(self):
        logger.info("Processing")

# Bad
class MyClass:
    def method(self):
        logger = get_logger(__name__)  # Created every call
        logger.info("Processing")
```

### 2. Structured Logging

Include relevant context in log messages:

```python
# Good
logger.info(f"Processed document", extra={
    'doc_id': doc_id,
    'chunks': len(chunks),
    'duration_ms': duration * 1000
})

# Also good - inline context
logger.info(f"Processed document doc_id={doc_id} chunks={len(chunks)} duration_ms={duration*1000:.2f}")

# Bad
logger.info("Processed document")
```

### 3. Performance Metrics

Use the dedicated performance logging function:

```python
from utils.logging_config import log_performance_metrics
import time

start = time.time()
result = expensive_operation()
duration = time.time() - start

log_performance_metrics(
    logger,
    'expensive_operation',
    duration,
    items_processed=len(result),
    cache_hits=cache.hits,
    cache_misses=cache.misses
)
```

### 4. Error Logging

Always include exception info for errors:

```python
# Good
try:
    risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # Handle error...

# Bad
except Exception as e:
    logger.error(f"Error: {e}")  # Missing traceback
```

### 5. Conditional Debug Logging

Avoid expensive operations in debug logs unless needed:

```python
# Good
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Large object state: {expensive_repr(obj)}")

# Bad
logger.debug(f"Large object state: {expensive_repr(obj)}")  # Always computed
```

## Configuration

### Environment Variables

- `LOG_LEVEL`: Set global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FILE`: Path to log file (optional)
- `LOG_COLOR`: Enable/disable colored output (true/false)

### Per-Module Configuration

```python
# In main script or configuration
from utils.logging_config import configure_module_loggers

# Reduce noise from verbose libraries
configure_module_loggers()

# Set specific module levels
logging.getLogger('database.db_manager').setLevel(logging.DEBUG)
logging.getLogger('embedding').setLevel(logging.WARNING)
```

### File Logging

```python
from utils.logging_config import configure_root_logger

# Setup with file and console output
configure_root_logger(
    level=logging.INFO,
    log_file='/var/log/customkb/app.log',
    console=True,
    colored=True
)
```

## Context Management

Add contextual information to all logs from a module:

```python
logger = get_logger(__name__, context={
    'kb_name': 'myproject',
    'version': '1.0.0'
})

# All logs will include kb_name and version
logger.info("Starting processing")  # Includes context automatically
```

## Performance Considerations

1. **Avoid String Formatting in Debug Logs**
   ```python
   # Good - formatted only if DEBUG enabled
   logger.debug("Processing %d items", len(items))
   
   # Bad - always formats string
   logger.debug(f"Processing {len(items)} items")
   ```

2. **Batch Log Writes**
   ```python
   # For high-frequency logging, consider batching
   from utils.logging_config import setup_file_handler
   handler = setup_file_handler(logger, 'batch.log')
   handler.setFormatter(logging.Formatter('%(message)s'))
   
   # Log in batches
   messages = []
   for item in large_dataset:
       if len(messages) >= 100:
           logger.info('\n'.join(messages))
           messages.clear()
       messages.append(f"Processed: {item}")
   ```

3. **Async Logging for I/O Heavy Operations**
   ```python
   import asyncio
   from concurrent.futures import ThreadPoolExecutor
   
   executor = ThreadPoolExecutor(max_workers=1)
   
   def log_async(logger, level, message):
       executor.submit(logger.log, level, message)
   ```

## Testing

Mock loggers in tests to verify logging behavior:

```python
import pytest
from unittest.mock import Mock, patch

def test_error_logging():
    with patch('mymodule.logger') as mock_logger:
        # Test function that should log error
        result = function_that_logs_error()
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "expected error message" in call_args
```

## Common Patterns

### Operation Timing

```python
from contextlib import contextmanager
import time

@contextmanager
def timed_operation(logger, operation_name):
    start = time.time()
    logger.debug(f"Starting {operation_name}")
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"Completed {operation_name} in {duration:.2f}s")

# Usage
with timed_operation(logger, "database_query"):
    results = db.query(sql)
```

### Progress Logging

```python
def process_items(items):
    total = len(items)
    log_interval = max(1, total // 10)  # Log every 10%
    
    for i, item in enumerate(items, 1):
        process(item)
        
        if i % log_interval == 0 or i == total:
            logger.info(f"Progress: {i}/{total} ({100*i/total:.1f}%)")
```

### Resource Usage Logging

```python
import psutil

def log_resource_usage(logger):
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    logger.info(f"Resources: memory={memory_mb:.1f}MB cpu={cpu_percent:.1f}%")
```

## Troubleshooting

### Debug Logging Not Showing

```python
# Check current log level
import logging
print(f"Root level: {logging.getLogger().level}")
print(f"Module level: {logger.level}")

# Force debug level
logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
```

### Too Much Output

```python
# Silence verbose modules
logging.getLogger('noisy_module').setLevel(logging.WARNING)

# Or use configuration
from utils.logging_config import configure_module_loggers
configure_module_loggers()
```

### Log Rotation

For production, use system log rotation or Python's RotatingFileHandler:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logger.addHandler(handler)
```

## Migration Checklist

When updating a module to use new logging standards:

- [ ] Import from `utils.logging_config` instead of `utils.logging_utils`
- [ ] Create logger at module level with `get_logger(__name__)`
- [ ] Replace bare `print()` statements with appropriate log levels
- [ ] Add `exc_info=True` to error logs with exceptions
- [ ] Include relevant context in log messages
- [ ] Use structured logging for metrics
- [ ] Test log output at different levels
- [ ] Update any log parsing scripts

## Examples by Module Type

### Database Module

```python
from utils.logging_config import get_logger, log_performance_metrics

logger = get_logger(__name__)

def execute_query(sql, params):
    logger.debug(f"Executing query: {sql[:100]}...")
    start = time.time()
    
    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        log_performance_metrics(
            logger, 'db_query',
            time.time() - start,
            rows_returned=len(rows),
            query_type=sql.split()[0]
        )
        
        return rows
    except sqlite3.Error as e:
        logger.error(f"Query failed: {e}", exc_info=True, extra={
            'sql': sql,
            'params': params
        })
        raise
```

### API Module

```python
from utils.logging_config import get_logger

logger = get_logger(__name__)

async def call_api(endpoint, data):
    logger.info(f"API call to {endpoint}")
    logger.debug(f"Request data: {data}")
    
    try:
        response = await client.post(endpoint, json=data)
        logger.info(f"API response: status={response.status_code}")
        return response.json()
    except RequestException as e:
        logger.error(f"API call failed: {e}", exc_info=True)
        raise
```

### Processing Module

```python
from utils.logging_config import get_logger

logger = get_logger(__name__)

def process_documents(documents):
    logger.info(f"Processing {len(documents)} documents")
    
    results = []
    errors = []
    
    for i, doc in enumerate(documents, 1):
        try:
            result = process_single(doc)
            results.append(result)
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(documents)} documents")
                
        except Exception as e:
            logger.warning(f"Failed to process document {doc['id']}: {e}")
            errors.append((doc['id'], str(e)))
    
    logger.info(f"Processing complete: success={len(results)} errors={len(errors)}")
    
    if errors:
        logger.error(f"Failed documents: {[e[0] for e in errors[:10]]}")
    
    return results, errors
```

#fin