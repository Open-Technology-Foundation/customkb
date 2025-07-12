# CustomKB Test Suite

Comprehensive testing framework for CustomKB components including unit tests, integration tests, and performance benchmarks.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python run_tests.py

# Run quick smoke test
python run_tests.py quick

# Run with coverage
python run_tests.py --coverage --html
```

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_config_manager.py
│   ├── test_db_manager.py
│   ├── test_embed_manager.py
│   ├── test_query_manager.py
│   ├── test_model_manager.py
│   └── utils/
│       ├── test_text_utils.py
│       └── test_logging_utils.py
├── integration/            # Integration tests for workflows
│   └── test_end_to_end.py
├── performance/            # Performance and scalability tests
│   └── test_performance.py
├── fixtures/               # Test data and utilities
│   └── mock_data.py
├── conftest.py            # Global test configuration
└── README.md
```

## Test Categories

### Unit Tests (`pytest -m unit`)
- **Config Manager**: Configuration loading, validation, environment overrides
- **Database Manager**: Text processing, chunking, metadata extraction
- **Embedding Manager**: Vector generation, caching, FAISS optimization
- **Query Manager**: Semantic search, context building, AI responses
- **Model Manager**: Model resolution, alias handling
- **Utilities**: Text processing, logging, file operations

### Integration Tests (`pytest -m integration`)
- **End-to-End Workflows**: Complete database → embed → query pipelines
- **CLI Integration**: Command-line interface testing
- **Configuration Integration**: Cross-component configuration validation
- **Real Data Integration**: Tests with wayang.net dataset (when available)

### Performance Tests (`pytest -m performance`)
- **Database Performance**: Large file processing, batch operations
- **Embedding Performance**: Batch processing, caching efficiency
- **Query Performance**: Vector search, concurrent operations
- **Memory Usage**: Resource consumption patterns
- **Scalability Limits**: Breaking point analysis

## Running Tests

### Basic Commands

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Fast tests (exclude slow performance tests)
pytest -m "not slow"

# Specific test file
pytest tests/unit/test_config_manager.py

# Specific test function
pytest tests/unit/test_config_manager.py::TestKnowledgeBase::test_init_with_kwargs -v
```

### Using the Test Runner

```bash
# Basic usage
python run_tests.py

# Specific test types
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance

# Safe mode (with memory limits and monitoring)
python run_tests.py --safe
python run_tests.py safe  # Shortcut

# Safe mode with options
python run_tests.py --safe --unit --memory-limit 1024  # 1GB limit
python run_tests.py --safe --timeout 60                # 60s timeout

# Fast tests only
python run_tests.py --fast

# With coverage
python run_tests.py --coverage --html

# Parallel execution
python run_tests.py --parallel 4

# Verbose output
python run_tests.py --verbose

# Install dependencies first
python run_tests.py --install-deps
```

### Advanced Options

```bash
# Run tests with specific markers
python run_tests.py --markers "unit and not slow"

# Run tests matching keyword
python run_tests.py --keyword "config"

# Run specific file
python run_tests.py --file tests/unit/test_db_manager.py

# CI/CD pipeline tests (exclude external dependencies)
python run_tests.py ci
```

## Test Markers

- `unit`: Unit tests (fast, isolated)
- `integration`: Integration tests (moderate speed)
- `performance`: Performance/benchmarking tests
- `slow`: Tests taking >5 seconds
- `requires_api`: Tests requiring real API keys
- `requires_data`: Tests requiring external test data

## Test Data

### Mock Data
- **Sample Configurations**: Various config file combinations
- **Sample Text Documents**: Text files for processing tests
- **Mock API Responses**: Realistic OpenAI/Anthropic responses
- **Database Fixtures**: Pre-populated test databases

### Real Data Integration
- **wayang.net Dataset**: Integration tests with real knowledge base
  - Location: `$VECTORDBS/wayang.net/`
  - Requires: wayang.net.cfg, wayang.net.db, wayang.net.faiss
  - Usage: `pytest -m requires_data`

## Coverage Requirements

- **Target Coverage**: 85% overall
- **Critical Paths**: 95% (database operations, API calls)
- **Error Handling**: All exception paths tested

### Generating Coverage Reports

```bash
# Terminal report
pytest --cov=. --cov-report=term-missing

# HTML report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# XML report (for CI)
pytest --cov=. --cov-report=xml
```

## Mocking Strategy

### External Dependencies
- **OpenAI/Anthropic APIs**: Mocked with realistic responses
- **FAISS Operations**: Mocked index operations
- **File System**: Temporary directories and files
- **Environment Variables**: Isolated test environments

### Test Isolation
- **Database Isolation**: Temporary SQLite databases
- **Configuration Isolation**: Temporary config files
- **Cache Isolation**: Temporary cache directories
- **Import Isolation**: Clean module state per test

## Performance Benchmarks

### Expected Performance Metrics
- **Database Processing**: >5 files/second for small files
- **Large File Processing**: <10 seconds for 250KB file
- **Memory Usage**: <100MB increase for typical operations
- **Vector Search**: <100ms regardless of index size
- **Cache Operations**: >100 embeddings/second

### Performance Test Categories
1. **Throughput Tests**: Files per second, chunks per second
2. **Latency Tests**: Response times for operations
3. **Memory Tests**: Memory usage patterns and cleanup
4. **Scalability Tests**: Performance with large datasets
5. **Concurrency Tests**: Performance under concurrent load

## Debugging Tests

### Verbose Output
```bash
# Show test names and results
pytest -v

# Show test output (print statements)
pytest -s

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb
```

### Test-Specific Debugging
```bash
# Run single test with debug info
pytest tests/unit/test_config_manager.py::TestKnowledgeBase::test_init_with_kwargs -v -s

# Run with specific log level
pytest --log-cli-level=DEBUG

# Run failed tests only
pytest --lf
```

## Continuous Integration

### CI Test Command
```bash
python run_tests.py ci
```

This runs tests suitable for CI/CD:
- Excludes tests requiring external APIs
- Excludes tests requiring large datasets
- Generates XML coverage report
- Uses shorter traceback format

### Required Environment Variables for CI
```bash
export OPENAI_API_KEY="sk-test..."  # Fake key for testing
export ANTHROPIC_API_KEY="sk-ant-test..."  # Fake key for testing
export VECTORDBS="/tmp/test_vectordbs"
export NLTK_DATA="/tmp/nltk_data"
```

## Contributing

### Adding New Tests
1. **Unit Tests**: Add to appropriate `test_*.py` file in `tests/unit/`
2. **Integration Tests**: Add to `tests/integration/test_end_to_end.py`
3. **Performance Tests**: Add to `tests/performance/test_performance.py`

### Test Naming Conventions
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<specific_behavior>`

### Test Documentation
- Include docstrings explaining test purpose
- Use descriptive test names
- Add comments for complex test logic
- Document any special setup requirements

### Mock Guidelines
- Mock external dependencies (APIs, file system)
- Use realistic mock data
- Isolate tests from each other
- Clean up resources in fixtures

---

For more information about CustomKB testing, see the main [CustomKB documentation](../README.md).