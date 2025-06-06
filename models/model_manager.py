#!/usr/bin/env python
"""
Model management for CustomKB.
Handles loading, validating, and resolving AI model configurations.
"""

import os
import json
from typing import Dict, Any, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_canonical_model(model_name: str) -> Dict[str, Any]:
  """
  Resolve a model name or alias to its canonical definition.
  
  Looks up model information from Models.json using exact matches,
  aliases, or partial matches in that order of preference.

  Args:
      model_name: The name or alias of the model.

  Returns:
      A dictionary containing model information and configuration.

  Raises:
      FileNotFoundError: If the Models.json file is not found.
      KeyError: If the model is not found in any form in Models.json.
  """
  script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  models_file = os.path.join(script_dir, 'Models.json')

  try:
    with open(models_file, 'r') as f:
      models = json.load(f)
  except FileNotFoundError:
    logger.error(f"Models.json file not found at {models_file}")
    raise FileNotFoundError(f"Models.json file not found at {models_file}")

  # Try direct lookup by model name
  if model_name in models:
    return models[model_name]

  # Try lookup by alias
  for model_id, model_info in models.items():
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
