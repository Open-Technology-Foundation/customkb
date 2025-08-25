#!/usr/bin/env python
"""
Custom exception classes for CustomKB.

This module defines specific exception types for different error scenarios,
improving error handling and debugging across the application.
"""

from typing import Optional, Any, Dict


class CustomKBError(Exception):
  """Base exception class for all CustomKB errors."""
  
  def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
    """
    Initialize CustomKB exception.
    
    Args:
        message: Error message
        details: Optional dictionary with additional error details
    """
    super().__init__(message)
    self.message = message
    self.details = details or {}
  
  def __str__(self):
    """String representation of the error."""
    if self.details:
      details_str = ', '.join([f"{k}={v}" for k, v in self.details.items()])
      return f"{self.message} ({details_str})"
    return self.message


# Configuration Errors
class ConfigurationError(CustomKBError):
  """Raised when there's an error in configuration."""
  pass


class KnowledgeBaseNotFoundError(ConfigurationError):
  """Raised when a knowledgebase cannot be found."""
  
  def __init__(self, kb_name: str, available_kbs: Optional[list] = None):
    """Initialize with KB name and available options."""
    message = f"Knowledgebase '{kb_name}' not found"
    details = {'kb_name': kb_name}
    
    if available_kbs:
      details['available'] = available_kbs
      message += f". Available: {', '.join(available_kbs)}"
    
    super().__init__(message, details)


class InvalidConfigurationError(ConfigurationError):
  """Raised when configuration values are invalid."""
  pass


# Database Errors
class DatabaseError(CustomKBError):
  """Base class for database-related errors."""
  pass


class ConnectionError(DatabaseError):
  """Raised when database connection fails."""
  pass


class QueryError(DatabaseError):
  """Raised when a database query fails."""
  
  def __init__(self, message: str, query: Optional[str] = None, params: Optional[tuple] = None):
    """Initialize with query details."""
    details = {}
    if query:
      details['query'] = query[:200]  # Truncate long queries
    if params:
      details['params'] = str(params)[:100]  # Truncate long params
    super().__init__(message, details)


class IndexError(DatabaseError):
  """Raised when there's an issue with database indexes."""
  pass


class TableNotFoundError(DatabaseError):
  """Raised when a required table doesn't exist."""
  
  def __init__(self, table_name: str):
    """Initialize with table name."""
    super().__init__(
      f"Table '{table_name}' not found",
      {'table': table_name}
    )


# Embedding Errors
class EmbeddingError(CustomKBError):
  """Base class for embedding-related errors."""
  pass


class ModelNotAvailableError(EmbeddingError):
  """Raised when an embedding model is not available."""
  
  def __init__(self, model_name: str, reason: Optional[str] = None):
    """Initialize with model details."""
    message = f"Model '{model_name}' is not available"
    if reason:
      message += f": {reason}"
    super().__init__(message, {'model': model_name})


class EmbeddingGenerationError(EmbeddingError):
  """Raised when embedding generation fails."""
  pass


class CacheError(EmbeddingError):
  """Raised when there's an issue with the embedding cache."""
  pass


# API Errors
class APIError(CustomKBError):
  """Base class for API-related errors."""
  pass


class AuthenticationError(APIError):
  """Raised when API authentication fails."""
  
  def __init__(self, service: str):
    """Initialize with service name."""
    super().__init__(
      f"Authentication failed for {service}",
      {'service': service}
    )


class RateLimitError(APIError):
  """Raised when API rate limit is exceeded."""
  
  def __init__(self, service: str, retry_after: Optional[int] = None):
    """Initialize with rate limit details."""
    message = f"Rate limit exceeded for {service}"
    details = {'service': service}
    
    if retry_after:
      message += f". Retry after {retry_after} seconds"
      details['retry_after'] = retry_after
    
    super().__init__(message, details)


class APIResponseError(APIError):
  """Raised when API returns an unexpected response."""
  
  def __init__(self, service: str, status_code: Optional[int] = None, response: Optional[str] = None):
    """Initialize with response details."""
    message = f"Unexpected response from {service}"
    details = {'service': service}
    
    if status_code:
      message += f" (status: {status_code})"
      details['status_code'] = status_code
    
    if response:
      details['response'] = response[:200]  # Truncate long responses
    
    super().__init__(message, details)


class ModelError(APIError):
  """Raised when there's an issue with AI model operations."""
  
  def __init__(self, model: str, reason: str):
    """Initialize with model details."""
    super().__init__(
      f"Model '{model}' error: {reason}",
      {'model': model, 'reason': reason}
    )


# Processing Errors
class ProcessingError(CustomKBError):
  """Base class for processing-related errors."""
  pass


class DocumentProcessingError(ProcessingError):
  """Raised when document processing fails."""
  
  def __init__(self, document: str, reason: str):
    """Initialize with document details."""
    super().__init__(
      f"Failed to process document '{document}': {reason}",
      {'document': document, 'reason': reason}
    )


class ChunkingError(ProcessingError):
  """Raised when text chunking fails."""
  pass


class BatchError(ProcessingError):
  """Raised when batch processing operations fail."""
  
  def __init__(self, batch_id: str, reason: str, failed_items: int = 0, total_items: int = 0):
    """Initialize with batch processing details."""
    message = f"Batch '{batch_id}' failed: {reason}"
    details = {
      'batch_id': batch_id,
      'reason': reason,
      'failed_items': failed_items,
      'total_items': total_items
    }
    
    if total_items > 0:
      success_rate = ((total_items - failed_items) / total_items) * 100
      message += f" (success rate: {success_rate:.1f}%)"
      details['success_rate'] = success_rate
    
    super().__init__(message, details)


class TokenLimitExceededError(ProcessingError):
  """Raised when token limit is exceeded."""
  
  def __init__(self, tokens: int, limit: int):
    """Initialize with token counts."""
    super().__init__(
      f"Token limit exceeded: {tokens} > {limit}",
      {'tokens': tokens, 'limit': limit}
    )


# Query Errors
class QueryProcessingError(CustomKBError):
  """Base class for query processing errors."""
  pass


class NoResultsError(QueryProcessingError):
  """Raised when a query returns no results."""
  
  def __init__(self, query: str):
    """Initialize with query."""
    super().__init__(
      f"No results found for query: {query[:100]}",
      {'query': query[:100]}
    )


class SearchError(QueryProcessingError):
  """Raised when search operation fails."""
  pass


# File System Errors
class FileSystemError(CustomKBError):
  """Base class for file system errors."""
  pass


class FileNotFoundError(FileSystemError):
  """Raised when a required file is not found."""
  
  def __init__(self, filepath: str):
    """Initialize with file path."""
    super().__init__(
      f"File not found: {filepath}",
      {'filepath': filepath}
    )


class PermissionError(FileSystemError):
  """Raised when there's a permission issue."""
  
  def __init__(self, filepath: str, operation: str):
    """Initialize with permission details."""
    super().__init__(
      f"Permission denied for {operation} on {filepath}",
      {'filepath': filepath, 'operation': operation}
    )


# Validation Errors
class ValidationError(CustomKBError):
  """Base class for validation errors."""
  pass


class InputValidationError(ValidationError):
  """Raised when input validation fails."""
  
  def __init__(self, field: str, value: Any, reason: str):
    """Initialize with validation details."""
    super().__init__(
      f"Invalid {field}: {reason}",
      {'field': field, 'value': str(value)[:100], 'reason': reason}
    )


class SecurityValidationError(ValidationError):
  """Raised when security validation fails."""
  
  def __init__(self, reason: str):
    """Initialize with security concern."""
    super().__init__(f"Security validation failed: {reason}")


# Resource Errors
class ResourceError(CustomKBError):
  """Base class for resource-related errors."""
  pass


class MemoryError(ResourceError):
  """Raised when memory limits are exceeded."""
  
  def __init__(self, required: int, available: int):
    """Initialize with memory details."""
    super().__init__(
      f"Insufficient memory: required {required}MB, available {available}MB",
      {'required': required, 'available': available}
    )


class DiskSpaceError(ResourceError):
  """Raised when disk space is insufficient."""
  
  def __init__(self, required: int, available: int):
    """Initialize with disk space details."""
    super().__init__(
      f"Insufficient disk space: required {required}MB, available {available}MB",
      {'required': required, 'available': available}
    )


# Retry and Recovery
class RetryableError(CustomKBError):
  """Base class for errors that can be retried."""
  
  def __init__(self, message: str, retry_count: int = 0, max_retries: int = 3):
    """Initialize with retry information."""
    super().__init__(
      message,
      {'retry_count': retry_count, 'max_retries': max_retries}
    )
    self.retry_count = retry_count
    self.max_retries = max_retries
  
  def can_retry(self) -> bool:
    """Check if operation can be retried."""
    return self.retry_count < self.max_retries


class TemporaryError(RetryableError):
  """Raised for temporary errors that should be retried."""
  pass


class PermanentError(CustomKBError):
  """Raised for permanent errors that should not be retried."""
  pass


# Utility functions
def handle_exception(e: Exception, logger=None, raise_custom: bool = True) -> Optional[CustomKBError]:
  """
  Convert standard exceptions to CustomKB exceptions.
  
  Args:
      e: The exception to handle
      logger: Optional logger for error logging
      raise_custom: Whether to raise the custom exception
      
  Returns:
      CustomKBError instance or None
      
  Raises:
      CustomKBError if raise_custom is True
  """
  import sqlite3
  import psutil
  
  custom_error = None
  
  # Map standard exceptions to custom ones
  if isinstance(e, sqlite3.DatabaseError):
    custom_error = DatabaseError(f"Database error: {e}")
  elif isinstance(e, sqlite3.IntegrityError):
    custom_error = DatabaseError(f"Database integrity error: {e}")
  elif isinstance(e, FileNotFoundError):
    custom_error = FileNotFoundError(str(e))
  elif isinstance(e, PermissionError):
    custom_error = PermissionError(str(e), "access")
  elif isinstance(e, MemoryError):
    custom_error = MemoryError(0, 0)  # Would need actual values
  elif isinstance(e, ValueError):
    custom_error = ValidationError(f"Validation error: {e}")
  elif isinstance(e, KeyError):
    custom_error = ConfigurationError(f"Missing configuration: {e}")
  elif isinstance(e, ConnectionError):
    custom_error = ConnectionError(f"Connection failed: {e}")
  elif isinstance(e, TimeoutError):
    custom_error = TemporaryError(f"Operation timed out: {e}")
  else:
    custom_error = CustomKBError(f"Unexpected error: {e}")
  
  # Log if logger provided
  if logger:
    logger.error(f"{custom_error.__class__.__name__}: {custom_error}", exc_info=True)
  
  # Raise if requested
  if raise_custom:
    raise custom_error from e
  
  return custom_error


#fin