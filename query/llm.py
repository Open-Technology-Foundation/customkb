"""
LiteLLM-based response generation for CustomKB queries.

Provides a unified interface for generating LLM responses across all
supported providers (OpenAI, Anthropic, Google, xAI, Ollama) via LiteLLM.
Replaces the hand-rolled multi-provider client management in response.py.
"""

from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

from models.model_manager import get_canonical_model
from query.prompt_templates import get_prompt_template
from utils.exceptions import APIError, ModelError
from utils.logging_config import get_logger

# Load .env from customkb directory
_env_path = Path(__file__).resolve().parent.parent / '.env'
if _env_path.exists():
  load_dotenv(_env_path)

logger = get_logger(__name__)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


def _to_litellm_model(model_name: str, provider: str) -> str:
  """Convert a model name + provider to LiteLLM's expected format.

  LiteLLM auto-detects OpenAI and Anthropic models by name prefix.
  Other providers need an explicit prefix (e.g., 'gemini/', 'xai/', 'ollama/').

  Args:
    model_name: The canonical model name from Models.json.
    provider: Provider string ('openai', 'anthropic', 'google', 'xai', 'ollama', 'local').

  Returns:
    Model string in LiteLLM format.
  """
  # Already has a provider prefix — pass through
  if '/' in model_name:
    return model_name

  match provider:
    case 'google':
      return f'gemini/{model_name}'
    case 'xai':
      return f'xai/{model_name}'
    case 'ollama' | 'local':
      return f'ollama/{model_name}'
    case _:
      # OpenAI and Anthropic are auto-detected by LiteLLM
      return model_name


def _get_provider_from_model_info(model_info: dict[str, Any]) -> str:
  """Derive API provider from model info fields.

  Mirrors the logic in response.py but kept here to avoid circular imports
  once response.py is replaced.

  Args:
    model_info: Dictionary from Models.json.

  Returns:
    Provider name string.
  """
  if 'provider' in model_info:
    return model_info['provider']

  parent = model_info.get('parent', '').lower()
  family = model_info.get('family', '').lower()

  # Ollama/local family overrides parent detection
  if 'ollama' in family:
    return 'ollama'
  if 'local' in parent:
    return 'local'
  if 'anthropic' in parent or 'claude' in family:
    return 'anthropic'
  if 'google' in parent or 'gemini' in family:
    return 'google'
  if 'xai' in parent or 'grok' in family:
    return 'xai'
  return 'openai'


async def get_response(
  messages: list[dict[str, str]],
  model: str,
  temperature: float = 0.7,
  max_tokens: int = 2000,
  provider: str | None = None,
) -> str:
  """Generate a response using LiteLLM's unified API.

  Args:
    messages: List of message dicts with 'role' and 'content' keys.
    model: Model name (canonical from Models.json or LiteLLM format).
    temperature: Response randomness (0.0-2.0).
    max_tokens: Maximum response tokens.
    provider: Optional provider hint. If None, derived from model name.

  Returns:
    Generated response text.

  Raises:
    APIError: If the LLM call fails.
  """
  litellm_model = _to_litellm_model(model, provider or '')

  logger.info(f"LiteLLM request: model={litellm_model}, temp={temperature}, max_tokens={max_tokens}")

  try:
    response = await litellm.acompletion(
      model=litellm_model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
      raise APIError(f"Empty response from model {litellm_model}")
    logger.info(f"LiteLLM response: {len(content)} characters")
    return content

  except litellm.AuthenticationError as e:
    logger.error(f"Authentication failed for {litellm_model}: {e}")
    raise APIError(f"Authentication failed for {litellm_model}: {e}") from e
  except litellm.RateLimitError as e:
    logger.error(f"Rate limit for {litellm_model}: {e}")
    raise APIError(f"Rate limit exceeded for {litellm_model}: {e}") from e
  except litellm.APIConnectionError as e:
    logger.error(f"Connection error for {litellm_model}: {e}")
    raise APIError(f"Connection error for {litellm_model}: {e}") from e
  except litellm.APIError as e:
    logger.error(f"API error for {litellm_model}: {e}")
    raise APIError(f"LiteLLM API error for {litellm_model}: {e}") from e


async def generate_ai_response(
  kb: Any,
  reference_string: str,
  query_text: str,
  prompt_template: str | None = None,
) -> str:
  """Generate an AI response using the KB's configured model via LiteLLM.

  Drop-in replacement for response.generate_ai_response(). Accepts the same
  arguments and returns the same type.

  Args:
    kb: KnowledgeBase instance (or any object with query_model, query_role, etc.).
    reference_string: Context from search results.
    query_text: Original query text.
    prompt_template: Optional prompt template name override.

  Returns:
    Generated response text.

  Raises:
    ModelError: If response generation fails.
  """
  model_name = kb.query_model
  try:
    model_info = get_canonical_model(model_name)
    model_name = model_info['model']
    provider = _get_provider_from_model_info(model_info)

    # Resolve prompt template
    template_name = prompt_template or getattr(kb, 'query_prompt_template', 'default')
    custom_role = getattr(kb, 'query_role', None)
    template = get_prompt_template(template_name, custom_role)

    # Build messages
    user_prompt = template['user'].format(
      reference_string=reference_string,
      query_text=query_text,
      # Legacy format vars used in response.py's duplicate templates
      context=reference_string,
      query=query_text,
    )
    messages = [
      {'role': 'system', 'content': template['system']},
      {'role': 'user', 'content': user_prompt},
    ]

    temperature = getattr(kb, 'query_temperature', 0.7)
    max_tokens = getattr(kb, 'query_max_tokens', 2000)

    response = await get_response(
      messages=messages,
      model=model_name,
      temperature=temperature,
      max_tokens=max_tokens,
      provider=provider,
    )

    if not response:
      raise ModelError(model_name, "Empty response from AI model")

    return response

  except ModelError:
    raise
  except APIError:
    raise
  except (FileNotFoundError, KeyError, ValueError) as e:
    logger.error(f"Model resolution failed for {model_name}: {e}")
    raise ModelError(model_name, f"Model resolution failed: {e}") from e


#fin
