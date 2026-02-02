"""
LiteLLM-based embedding provider for CustomKB.

Provides a unified embedding interface via LiteLLM, replacing the hand-rolled
OpenAI and Google provider classes in providers.py. Local SentenceTransformer
models are delegated back to the existing SentenceTransformerProvider since
LiteLLM does not support local models.
"""


from pathlib import Path

import litellm
from dotenv import load_dotenv

from utils.exceptions import APIError
from utils.logging_config import get_logger

# Load .env from customkb directory
_env_path = Path(__file__).resolve().parent.parent / '.env'
if _env_path.exists():
  load_dotenv(_env_path)

logger = get_logger(__name__)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True

# Models that require local SentenceTransformer (not supported by LiteLLM)
LOCAL_MODELS = {'bge-m3', 'all-minilm-l6-v2'}


def _to_litellm_embedding_model(model: str) -> str:
  """Convert a model name to LiteLLM's expected embedding format.

  LiteLLM auto-detects OpenAI embedding models by name prefix.
  Google models need a 'gemini/' prefix.

  Args:
    model: The embedding model name.

  Returns:
    Model string in LiteLLM format.
  """
  if '/' in model:
    return model

  model_lower = model.lower()
  if 'gemini' in model_lower or 'google' in model_lower:
    return f'gemini/{model}'

  # OpenAI text-embedding-* models are auto-detected
  return model


async def get_embeddings(
  texts: list[str],
  model: str,
  dimensions: int | None = None,
) -> list[list[float]]:
  """Get embeddings for a list of texts via LiteLLM.

  For API-based models (OpenAI, Google), uses LiteLLM's unified interface.
  For local models (bge-m3, all-minilm-l6-v2), delegates to
  SentenceTransformerProvider.

  Args:
    texts: List of texts to embed.
    model: Embedding model name.
    dimensions: Optional output dimensions (supported by some models).

  Returns:
    List of embedding vectors.

  Raises:
    APIError: If the embedding call fails.
  """
  model_lower = model.lower()

  # Delegate local models to SentenceTransformerProvider
  if model_lower in LOCAL_MODELS or model_lower.startswith('sentence-transformers/'):
    from embedding.providers import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model)
    return await provider.get_embeddings(texts, model)

  litellm_model = _to_litellm_embedding_model(model)
  logger.info(f"LiteLLM embedding: model={litellm_model}, texts={len(texts)}")

  try:
    response = await litellm.aembedding(
      model=litellm_model,
      input=texts,
      **({"dimensions": dimensions} if dimensions else {}),
    )
    embeddings = [item['embedding'] for item in response.data]
    logger.info(f"LiteLLM embedding: {len(embeddings)} vectors, dims={len(embeddings[0]) if embeddings else 0}")
    return embeddings

  except litellm.AuthenticationError as e:
    logger.error(f"Embedding auth failed for {litellm_model}: {e}")
    raise APIError(f"Authentication failed for embedding model {litellm_model}: {e}") from e
  except litellm.RateLimitError as e:
    logger.error(f"Embedding rate limit for {litellm_model}: {e}")
    raise APIError(f"Rate limit exceeded for embedding model {litellm_model}: {e}") from e
  except litellm.APIConnectionError as e:
    logger.error(f"Embedding connection error for {litellm_model}: {e}")
    raise APIError(f"Connection error for embedding model {litellm_model}: {e}") from e
  except litellm.APIError as e:
    logger.error(f"Embedding API error for {litellm_model}: {e}")
    raise APIError(f"LiteLLM embedding error for {litellm_model}: {e}") from e


def get_embedding_sync(text: str, model: str) -> list[float]:
  """Get a single embedding synchronously.

  Convenience wrapper for code paths that can't use async.

  Args:
    text: Text to embed.
    model: Embedding model name.

  Returns:
    Embedding vector.

  Raises:
    APIError: If the embedding call fails.
  """
  model_lower = model.lower()

  # Delegate local models
  if model_lower in LOCAL_MODELS or model_lower.startswith('sentence-transformers/'):
    from embedding.providers import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model)
    return provider.get_embedding_sync(text, model)

  litellm_model = _to_litellm_embedding_model(model)

  try:
    response = litellm.embedding(
      model=litellm_model,
      input=[text],
    )
    return response.data[0]['embedding']

  except (litellm.AuthenticationError, litellm.RateLimitError,
          litellm.APIConnectionError, litellm.APIError) as e:
    logger.error(f"Sync embedding error for {litellm_model}: {e}")
    raise APIError(f"Embedding failed for {litellm_model}: {e}") from e


def get_embedding_dimensions(model: str) -> int:
  """Get the output dimensions for an embedding model.

  Args:
    model: Model name.

  Returns:
    Number of dimensions.
  """
  dimensions = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'gemini-embedding-001': 768,
    'bge-m3': 1024,
    'all-minilm-l6-v2': 384,
  }
  return dimensions.get(model.lower(), 1536)


#fin
