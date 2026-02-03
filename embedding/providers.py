#!/usr/bin/env python
"""
Embedding provider clients for CustomKB.

This module handles different embedding providers (OpenAI, Google AI)
with unified interface and error handling.
"""

import asyncio
import os
import threading
from typing import Any

from openai import AsyncOpenAI, OpenAI

from utils.exceptions import APIError, AuthenticationError
from utils.logging_config import get_logger
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

  async def get_embeddings(self, texts: list[str], model: str) -> list[list[float]]:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model: Model name to use

    Returns:
        List of embedding vectors
    """
    raise NotImplementedError("Subclasses must implement get_embeddings")

  def get_embedding_sync(self, text: str, model: str) -> list[float]:
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

  async def get_embeddings(self, texts: list[str], model: str) -> list[list[float]]:
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

    except TimeoutError as e:
      logger.error(f"OpenAI API timeout: {e}")
      raise APIError("Failed to get OpenAI embeddings: Request timed out after 300 seconds") from e
    except (ValueError, AttributeError, RuntimeError, OSError) as e:
      logger.error(f"OpenAI API error: {e}")
      raise APIError(f"Failed to get OpenAI embeddings: {e}") from e

  def get_embedding_sync(self, text: str, model: str) -> list[float]:
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

    except TimeoutError as e:
      logger.error(f"OpenAI sync API timeout: {e}")
      raise APIError("Failed to get OpenAI embedding: Request timed out after 300 seconds") from e
    except (ValueError, AttributeError, RuntimeError, OSError) as e:
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
    except (ValueError, RuntimeError, OSError) as e:
      logger.error(f"Failed to initialize Google AI client: {e}")
      raise AuthenticationError(f"Google AI initialization failed: {e}") from e

  async def get_embeddings(self, texts: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
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

    except (ValueError, AttributeError, RuntimeError, OSError) as e:
      logger.error(f"Google AI API error: {e}")
      raise APIError(f"Failed to get Google embeddings: {e}") from e

  async def _get_single_embedding_async(self, text: str, model: str) -> list[float]:
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

  def get_embedding_sync(self, text: str, model: str = "gemini-embedding-001") -> list[float]:
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

    except (ValueError, AttributeError, KeyError, RuntimeError) as e:
      logger.error(f"Google AI sync API error: {e}")
      raise APIError(f"Failed to get Google embedding: {e}") from e


class SentenceTransformerProvider(EmbeddingProvider):
  """Local embedding provider using sentence-transformers.

  Supports BAAI/bge-m3 (1024 dims) and all-MiniLM-L6-v2 (384 dims) models.
  No API key required - runs entirely locally.
  """

  SUPPORTED_MODELS = {
    'bge-m3': 'BAAI/bge-m3',
    'all-minilm-l6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
  }

  def __init__(self, model_name: str = 'bge-m3'):
    """
    Initialize SentenceTransformer provider.

    Args:
        model_name: Local model name (bge-m3, all-minilm-l6-v2, or HuggingFace path)
    """
    super().__init__(api_key=None)  # No API key needed
    self.model_name = model_name.lower()
    self._model = None
    self._model_lock = threading.Lock()

  def _load_model(self):
    """Lazy load the sentence transformer model with thread-safe locking."""
    if self._model is None:
      with self._model_lock:
        if self._model is None:  # Double-check after acquiring lock
          from sentence_transformers import SentenceTransformer
          hf_model = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
          logger.info(f"Loading local embedding model: {hf_model}")
          self._model = SentenceTransformer(hf_model, device="cpu")
          logger.info(f"Model loaded: {hf_model}")
    return self._model

  async def get_embeddings(self, texts: list[str], model: str) -> list[list[float]]:
    """
    Get embeddings using local sentence-transformers model.

    Args:
        texts: List of texts to embed
        model: Model name (used for logging, actual model set in __init__)

    Returns:
        List of embedding vectors
    """
    try:
      # Run in thread pool to avoid blocking the event loop
      loop = asyncio.get_event_loop()
      return await loop.run_in_executor(None, self._embed_sync, texts)
    except (RuntimeError, ValueError, ImportError, OSError) as e:
      logger.error(f"SentenceTransformer embedding error: {e}")
      raise APIError(f"Failed to get local embeddings: {e}") from e

  def _embed_sync(self, texts: list[str]) -> list[list[float]]:
    """
    Synchronous embedding generation.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    model = self._load_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

  def get_embedding_sync(self, text: str, model: str) -> list[float]:
    """
    Get single embedding (synchronous).

    Args:
        text: Text to embed
        model: Model name (unused, model set in __init__)

    Returns:
        Embedding vector
    """
    try:
      return self._embed_sync([text])[0]
    except (RuntimeError, ValueError, ImportError, IndexError) as e:
      logger.error(f"SentenceTransformer sync embedding error: {e}")
      raise APIError(f"Failed to get local embedding: {e}") from e


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
    model_lower = model.lower()

    # Determine provider from model name
    if 'gemini' in model_lower or 'google' in model_lower:
      provider_type = 'google'
    elif model_lower in ('bge-m3', 'all-minilm-l6-v2') or model_lower.startswith('sentence-transformers/'):
      provider_type = 'local'
    else:
      # Default to OpenAI for other models
      provider_type = 'openai'

    # Cache providers - include model name for local providers
    if provider_type == 'local':
      cache_key = f"{provider_type}_{model_lower}"
    else:
      cache_key = f"{provider_type}:{api_key[:10] if api_key else 'env'}"

    if cache_key not in cls._providers:
      if provider_type == 'google':
        cls._providers[cache_key] = GoogleAIProvider(api_key)
      elif provider_type == 'local':
        cls._providers[cache_key] = SentenceTransformerProvider(model)
      else:
        cls._providers[cache_key] = OpenAIProvider(api_key)

    return cls._providers[cache_key]

  @classmethod
  def clear_cache(cls):
    """Clear cached providers."""
    cls._providers.clear()


async def get_embeddings_with_provider(texts: list[str], model: str,
                                      kb: Any = None) -> list[list[float]]:
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
    'bge-m3': 1024,
    'all-minilm-l6-v2': 384,
  }

  return dimensions.get(model.lower(), 1536)  # Default to 1536


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
    'gemini-embedding-001',
    'bge-m3',
    'all-minilm-l6-v2',
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
    'google': 'gemini-embedding-001',
    'bge': 'bge-m3',
    'minilm': 'all-minilm-l6-v2',
  }

  if model_lower in aliases:
    model = aliases[model_lower]

  # Allow sentence-transformers/ prefixed models
  if model_lower.startswith('sentence-transformers/'):
    return model

  # Check if model is supported
  if model.lower() not in [m.lower() for m in supported_models]:
    default_model = 'text-embedding-3-small'
    if kb:
      default_model = getattr(kb, 'default_embedding_model', default_model)

    logger.warning(f"Unknown embedding model: {model}, using {default_model}")
    model = default_model

  return model


#fin
