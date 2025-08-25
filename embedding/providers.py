#!/usr/bin/env python
"""
Embedding provider clients for CustomKB.

This module handles different embedding providers (OpenAI, Google AI)
with unified interface and error handling.
"""

import os
import sys
import asyncio
from typing import List, Optional, Dict, Any
import time

from openai import OpenAI, AsyncOpenAI
from utils.logging_config import get_logger
from utils.exceptions import APIError, AuthenticationError
from utils.security_utils import validate_api_key

logger = get_logger(__name__)

# Try to import Google AI
try:
  from google import genai
  GOOGLE_AI_AVAILABLE = True
except ImportError:
  GOOGLE_AI_AVAILABLE = False
  logger.debug("Google AI library not available")


class EmbeddingProvider:
  """Base class for embedding providers."""
  
  def __init__(self, api_key: str = None):
    """
    Initialize provider.
    
    Args:
        api_key: API key for the provider
    """
    self.api_key = api_key
    self.client = None
    self.async_client = None
  
  async def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
    """
    Get embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model: Model name to use
        
    Returns:
        List of embedding vectors
    """
    raise NotImplementedError("Subclasses must implement get_embeddings")
  
  def get_embedding_sync(self, text: str, model: str) -> List[float]:
    """
    Get embedding for a single text (synchronous).
    
    Args:
        text: Text to embed
        model: Model name to use
        
    Returns:
        Embedding vector
    """
    raise NotImplementedError("Subclasses must implement get_embedding_sync")


class OpenAIProvider(EmbeddingProvider):
  """OpenAI embedding provider."""
  
  def __init__(self, api_key: str = None):
    """
    Initialize OpenAI provider.
    
    Args:
        api_key: OpenAI API key
    """
    super().__init__(api_key)
    
    # Get API key from environment if not provided
    if not api_key:
      api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
      raise AuthenticationError("OpenAI API key not provided")
    
    # Validate API key format
    if not validate_api_key(api_key, 'sk-', 40):
      raise AuthenticationError("Invalid OpenAI API key format")
    
    self.api_key = api_key
    # Add timeout settings to prevent hanging on slow requests
    # 60 seconds for connection, 300 seconds for read
    import httpx
    timeout = httpx.Timeout(60.0, read=300.0)
    self.client = OpenAI(api_key=api_key, timeout=timeout)
    self.async_client = AsyncOpenAI(api_key=api_key, timeout=timeout)
  
  async def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
    """
    Get embeddings from OpenAI.
    
    Args:
        texts: List of texts to embed
        model: OpenAI model name
        
    Returns:
        List of embedding vectors
    """
    try:
      # Validate model name
      valid_models = ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']
      if model not in valid_models:
        logger.warning(f"Unknown OpenAI model: {model}, using text-embedding-3-small")
        model = 'text-embedding-3-small'
      
      # Make API call
      response = await self.async_client.embeddings.create(
        input=texts,
        model=model
      )
      
      # Extract embeddings
      embeddings = [item.embedding for item in response.data]
      
      return embeddings
      
    except asyncio.TimeoutError as e:
      logger.error(f"OpenAI API timeout: {e}")
      raise APIError(f"Failed to get OpenAI embeddings: Request timed out after 300 seconds") from e
    except Exception as e:
      logger.error(f"OpenAI API error: {e}")
      raise APIError(f"Failed to get OpenAI embeddings: {e}") from e
  
  def get_embedding_sync(self, text: str, model: str) -> List[float]:
    """
    Get single embedding from OpenAI (synchronous).
    
    Args:
        text: Text to embed
        model: OpenAI model name
        
    Returns:
        Embedding vector
    """
    try:
      response = self.client.embeddings.create(
        input=[text],
        model=model
      )
      return response.data[0].embedding
      
    except asyncio.TimeoutError as e:
      logger.error(f"OpenAI sync API timeout: {e}")
      raise APIError(f"Failed to get OpenAI embedding: Request timed out after 300 seconds") from e
    except Exception as e:
      logger.error(f"OpenAI sync API error: {e}")
      raise APIError(f"Failed to get OpenAI embedding: {e}") from e


class GoogleAIProvider(EmbeddingProvider):
  """Google AI embedding provider."""
  
  def __init__(self, api_key: str = None):
    """
    Initialize Google AI provider.
    
    Args:
        api_key: Google API key
    """
    super().__init__(api_key)
    
    if not GOOGLE_AI_AVAILABLE:
      raise ImportError("Google AI library not installed. Install with: pip install google-generativeai")
    
    # Get API key from environment if not provided
    if not api_key:
      api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if not api_key:
      raise AuthenticationError("Google API key not provided")
    
    # Validate API key (basic check)
    if not validate_api_key(api_key, min_length=20):
      raise AuthenticationError("Invalid Google API key format")
    
    self.api_key = api_key
    
    # Initialize client
    try:
      self.client = genai.Client(api_key=api_key)
      logger.info("Google AI client initialized successfully")
    except Exception as e:
      logger.error(f"Failed to initialize Google AI client: {e}")
      raise AuthenticationError(f"Google AI initialization failed: {e}") from e
  
  async def get_embeddings(self, texts: List[str], model: str = "gemini-embedding-001") -> List[List[float]]:
    """
    Get embeddings from Google AI.
    
    Args:
        texts: List of texts to embed
        model: Google model name (default: gemini-embedding-001)
        
    Returns:
        List of embedding vectors
    """
    try:
      embeddings = []
      
      # Google AI typically processes one at a time
      for text in texts:
        response = await self._get_single_embedding_async(text, model)
        embeddings.append(response)
      
      return embeddings
      
    except Exception as e:
      logger.error(f"Google AI API error: {e}")
      raise APIError(f"Failed to get Google embeddings: {e}") from e
  
  async def _get_single_embedding_async(self, text: str, model: str) -> List[float]:
    """
    Get single embedding from Google AI (async).
    
    Args:
        text: Text to embed
        model: Model name
        
    Returns:
        Embedding vector
    """
    # Google AI client might not have async support yet
    # Use sync version in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.get_embedding_sync, text, model)
  
  def get_embedding_sync(self, text: str, model: str = "gemini-embedding-001") -> List[float]:
    """
    Get single embedding from Google AI (synchronous).
    
    Args:
        text: Text to embed
        model: Google model name
        
    Returns:
        Embedding vector
    """
    try:
      # Use the appropriate method based on the library version
      if hasattr(self.client, 'embed'):
        response = self.client.embed(
          model=model,
          text=text
        )
        return response.embedding
      else:
        # Alternative method if API changes
        response = self.client.models.embed_content(
          model=f"models/{model}",
          content=text
        )
        return response['embedding']
      
    except Exception as e:
      logger.error(f"Google AI sync API error: {e}")
      raise APIError(f"Failed to get Google embedding: {e}") from e


class EmbeddingProviderFactory:
  """Factory for creating embedding providers."""
  
  _providers = {}
  
  @classmethod
  def get_provider(cls, model: str, api_key: str = None) -> EmbeddingProvider:
    """
    Get appropriate provider for a model.
    
    Args:
        model: Model name
        api_key: Optional API key
        
    Returns:
        EmbeddingProvider instance
    """
    # Determine provider from model name
    if 'gemini' in model.lower() or 'google' in model.lower():
      provider_type = 'google'
    else:
      # Default to OpenAI for other models
      provider_type = 'openai'
    
    # Cache providers
    cache_key = f"{provider_type}:{api_key[:10] if api_key else 'env'}"
    
    if cache_key not in cls._providers:
      if provider_type == 'google':
        cls._providers[cache_key] = GoogleAIProvider(api_key)
      else:
        cls._providers[cache_key] = OpenAIProvider(api_key)
    
    return cls._providers[cache_key]
  
  @classmethod
  def clear_cache(cls):
    """Clear cached providers."""
    cls._providers.clear()


async def get_embeddings_with_provider(texts: List[str], model: str, 
                                      kb: Any = None) -> List[List[float]]:
  """
  Get embeddings using the appropriate provider.
  
  Args:
      texts: List of texts to embed
      model: Model name
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      List of embedding vectors
  """
  # Get provider
  provider = EmbeddingProviderFactory.get_provider(model)
  
  # Apply rate limiting if configured
  if kb:
    rate_limit_delay = getattr(kb, 'embedding_rate_limit_delay', 0.1)
    if rate_limit_delay > 0:
      await asyncio.sleep(rate_limit_delay)
  
  # Get embeddings
  embeddings = await provider.get_embeddings(texts, model)
  
  return embeddings


def get_embedding_dimensions(model: str) -> int:
  """
  Get the output dimensions for an embedding model.
  
  Args:
      model: Model name
      
  Returns:
      Number of dimensions
  """
  dimensions = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'gemini-embedding-001': 768,  # Default, can be 768/1536/3072
  }
  
  return dimensions.get(model, 1536)  # Default to 1536


def validate_model_name(model: str, kb: Any = None) -> str:
  """
  Validate and normalize model name.
  
  Args:
      model: Model name to validate
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      Validated model name
  """
  # Check if model is supported
  supported_models = [
    'text-embedding-ada-002',
    'text-embedding-3-small', 
    'text-embedding-3-large',
    'gemini-embedding-001'
  ]
  
  # Normalize model name
  model_lower = model.lower()
  
  # Map common aliases
  aliases = {
    'ada': 'text-embedding-ada-002',
    'ada-002': 'text-embedding-ada-002',
    'small': 'text-embedding-3-small',
    'large': 'text-embedding-3-large',
    'gemini': 'gemini-embedding-001',
    'google': 'gemini-embedding-001'
  }
  
  if model_lower in aliases:
    model = aliases[model_lower]
  
  # Check if model is supported
  if model not in supported_models:
    default_model = 'text-embedding-3-small'
    if kb:
      default_model = getattr(kb, 'default_embedding_model', default_model)
    
    logger.warning(f"Unknown embedding model: {model}, using {default_model}")
    model = default_model
  
  return model


#fin