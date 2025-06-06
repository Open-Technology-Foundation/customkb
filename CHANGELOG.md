# Changelog

All notable changes to the CustomKB project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation overhaul:
  - Enhanced README.md with detailed usage examples and troubleshooting
  - Created DEVELOPMENT.md for contributor guidelines
  - Updated CLAUDE.md with AI assistant instructions
  - Added example.cfg configuration template
- Improved internal documentation:
  - Enhanced module and function docstrings
  - Added security-focused inline comments
  - Improved type hints and examples

### Fixed
- Fixed missing KnowledgeBase import in customkb.py that caused edit command to fail
- Enhanced help text with practical examples for all commands

## [0.1.1] - 2024-12-06

### Added
- Versioning system with semantic versioning support
- Build number tracking via git hooks
- Version command with --build flag
- Automatic version incrementing script (version.sh)

### Changed
- Updated version management to support MAJOR.MINOR.PATCH.BUILD format
- Enhanced git integration for automatic build number updates

## [0.1.0] - 2024-04-24

### Added
- Initial version of CustomKB
- Database processing capabilities for text files
- Embedding generation using OpenAI models  
- Semantic search with context-aware AI responses
- Configuration system with environment variable overrides
- Command-line interface for database, embed, query, and edit operations
- Multi-language support for text processing
- Security features including input validation and path sanitization
- Performance optimizations:
  - Two-tier caching system (memory + disk)
  - Batch processing for API calls
  - Checkpoint saving for resilience
  - Adaptive FAISS index selection
- Support for multiple file formats:
  - Markdown (.md)
  - HTML (.html, .htm)
  - Code files (Python, JavaScript, Java, etc.)
  - Plain text files
- Advanced configuration system with 5 sections:
  - DEFAULT: Core settings
  - API: Rate limiting and concurrency
  - LIMITS: Resource constraints
  - PERFORMANCE: Optimization parameters
  - ALGORITHMS: Processing thresholds

### Security
- Input validation for all user-provided data
- Path traversal prevention
- API key format validation
- SQL injection protection
- Sensitive data masking in logs

### Performance
- Asynchronous API operations
- Configurable batch sizes
- Memory-efficient file processing
- Query result caching with TTL
- Optimized database indexing

#fin