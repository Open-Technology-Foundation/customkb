#!/usr/bin/env python
"""
Model management for CustomKB.
Handles loading, validating, and resolving AI model configurations.
"""

import json
import os
from typing import Any

from utils.logging_config import get_logger

# Add this module-level variable for test compatibility
models_file = os.path.join(os.path.dirname(__file__), "..", "Models.json")

# Module-level cache for parsed Models.json (performance optimization)
_models_cache: dict[str, Any] | None = None

logger = get_logger(__name__)

def get_canonical_model(model_name: str) -> dict[str, Any]:
  """
  Resolve a model name or alias to its canonical definition.

  Looks up model information from Models.json using exact matches,
  aliases, or partial matches in that order of preference.

  Args:
      model_name: The name or alias of the model.

  Returns:
      A dictionary containing model information and configuration.

  Raises:
      ValueError: If model_name is None, empty, or not a string.
      FileNotFoundError: If the Models.json file is not found.
      KeyError: If the model is not found in any form in Models.json.
  """
  # Validate input
  if not model_name or not isinstance(model_name, str):
    raise ValueError("model_name must be a non-empty string")

  model_name = model_name.strip()
  if not model_name:
    raise ValueError("model_name must be a non-empty string")

  # Check cache first (performance optimization)
  global _models_cache
  if _models_cache is not None:
    models = _models_cache
  else:
    # Use the module-level models_file variable (can be patched in tests)
    try:
      with open(models_file) as f:
        models = json.load(f)
        # Populate cache for future calls
        _models_cache = models
    except FileNotFoundError:
      logger.error(f"Models.json file not found at {models_file}")
      raise

  # Try direct lookup by model name
  if model_name in models:
    return models[model_name]

  # Try lookup by alias
  for _model_id, model_info in models.items():
    if model_info.get('alias') == model_name:
      return model_info

  # Fall back to partial match if no exact match found
  for model_id, model_info in models.items():
    if model_name in model_id or (model_info.get('alias') and model_name in model_info['alias']):
      if logger:
        logger.warning(f"Using partial match: {model_id} for {model_name}")
      return model_info

  # No matches found
  if logger:
    logger.error(f"Model {model_name} not found in Models.json")
  raise KeyError(f"Model {model_name} not found in Models.json")

#fin
