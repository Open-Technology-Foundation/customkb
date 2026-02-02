"""
Unit tests for model_manager.py
Tests model resolution, alias handling, and Models.json parsing.
"""

import json
import os
from unittest.mock import patch

import pytest

from models.model_manager import get_canonical_model


class TestGetCanonicalModel:
  """Test the get_canonical_model function."""

  def test_direct_model_lookup(self, temp_data_manager):
    """Test direct model name lookup."""
    models_data = {
      "gpt-4": {
        "model": "gpt-4",
        "provider": "openai",
        "type": "chat"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("gpt-4")

      assert result["model"] == "gpt-4"
      assert result["provider"] == "openai"
      assert result["type"] == "chat"

  def test_alias_lookup(self, temp_data_manager):
    """Test model lookup by alias."""
    models_data = {
      "gpt-4-turbo-preview": {
        "model": "gpt-4-turbo-preview",
        "alias": "gpt4",
        "provider": "openai",
        "type": "chat"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("gpt4")

      assert result["model"] == "gpt-4-turbo-preview"
      assert result["alias"] == "gpt4"
      assert result["provider"] == "openai"

  def test_partial_match_fallback(self, temp_data_manager):
    """Test partial match fallback when exact match not found."""
    models_data = {
      "text-embedding-3-large": {
        "model": "text-embedding-3-large",
        "provider": "openai",
        "type": "embedding"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      # Search for partial match
      result = get_canonical_model("embedding-3")

      assert result["model"] == "text-embedding-3-large"
      assert result["provider"] == "openai"
      assert result["type"] == "embedding"

  def test_partial_match_with_alias(self, temp_data_manager):
    """Test partial match with alias fallback."""
    models_data = {
      "claude-3-sonnet-20240229": {
        "model": "claude-3-sonnet-20240229",
        "alias": "claude-sonnet",
        "provider": "anthropic",
        "type": "chat"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      # Search for partial match in alias
      result = get_canonical_model("sonnet")

      assert result["model"] == "claude-3-sonnet-20240229"
      assert result["alias"] == "claude-sonnet"

  def test_model_not_found(self, temp_data_manager):
    """Test behavior when model is not found."""
    models_data = {
      "gpt-4": {
        "model": "gpt-4",
        "provider": "openai"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file), \
         pytest.raises(KeyError, match="nonexistent-model not found"):
      get_canonical_model("nonexistent-model")

  def test_models_file_not_found(self):
    """Test behavior when Models.json file is not found."""
    with patch('models.model_manager.models_file', "/nonexistent/Models.json"), \
         pytest.raises(FileNotFoundError, match="No such file or directory"):
      get_canonical_model("gpt-4")

  def test_invalid_json_file(self, temp_data_manager):
    """Test behavior with invalid JSON file."""
    # Create invalid JSON file
    temp_dir = temp_data_manager.create_temp_dir()
    models_file = os.path.join(temp_dir, "Models.json")
    with open(models_file, 'w') as f:
      f.write("invalid json content {{{")

    with patch('models.model_manager.models_file', models_file), pytest.raises(json.JSONDecodeError):
      get_canonical_model("gpt-4")

  def test_complex_model_data(self, temp_data_manager):
    """Test with complex model data including multiple fields."""
    models_data = {
      "gpt-4-vision-preview": {
        "model": "gpt-4-vision-preview",
        "alias": "gpt4v",
        "provider": "openai",
        "type": "chat",
        "capabilities": ["text", "vision"],
        "max_tokens": 4096,
        "temperature_range": [0.0, 2.0],
        "cost_per_1k_tokens": {
          "input": 0.01,
          "output": 0.03
        }
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("gpt4v")

      assert result["model"] == "gpt-4-vision-preview"
      assert result["capabilities"] == ["text", "vision"]
      assert result["max_tokens"] == 4096
      assert result["cost_per_1k_tokens"]["input"] == 0.01

  def test_multiple_partial_matches(self, temp_data_manager):
    """Test behavior with multiple partial matches (should return first)."""
    models_data = {
      "text-embedding-ada-002": {
        "model": "text-embedding-ada-002",
        "provider": "openai",
        "type": "embedding"
      },
      "text-embedding-3-small": {
        "model": "text-embedding-3-small",
        "provider": "openai",
        "type": "embedding"
      },
      "text-embedding-3-large": {
        "model": "text-embedding-3-large",
        "provider": "openai",
        "type": "embedding"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      # Search for partial match - should return first match found
      result = get_canonical_model("embedding")

      # Should return one of the embedding models
      assert "embedding" in result["model"]
      assert result["provider"] == "openai"
      assert result["type"] == "embedding"

  def test_case_sensitivity(self, temp_data_manager):
    """Test case sensitivity in model lookup."""
    models_data = {
      "GPT-4": {
        "model": "GPT-4",
        "provider": "openai"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      # Exact case match should work
      result = get_canonical_model("GPT-4")
      assert result["model"] == "GPT-4"

      # Different case should fail exact match but might work with partial
      with pytest.raises(KeyError):
        get_canonical_model("gpt-4")

  def test_empty_models_file(self, temp_data_manager):
    """Test behavior with empty models file."""
    models_data = {}

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file), pytest.raises(KeyError, match="test-model not found"):
      get_canonical_model("test-model")

  def test_models_with_missing_fields(self, temp_data_manager):
    """Test models with missing optional fields."""
    models_data = {
      "basic-model": {
        "model": "basic-model"
        # Missing provider, type, alias, etc.
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("basic-model")

      assert result["model"] == "basic-model"
      assert result.get("provider") is None
      assert result.get("alias") is None

  def test_models_file_path_resolution(self):
    """Test that models file path is correctly resolved relative to script."""
    from models import model_manager

    # Verify the models_file path exists and points to a valid location
    # (either the actual file or a path ending with Models.json)
    assert model_manager.models_file.endswith('Models.json')

    # The path should be absolute or relative to the models directory
    import os
    models_dir = os.path.dirname(model_manager.__file__)
    expected_path = os.path.join(models_dir, "..", "Models.json")
    assert model_manager.models_file == expected_path

  def _create_models_file(self, temp_data_manager, models_data):
    """Helper to create a temporary Models.json file."""
    temp_dir = temp_data_manager.create_temp_dir()
    models_file = os.path.join(temp_dir, "Models.json")

    with open(models_file, 'w') as f:
      json.dump(models_data, f, indent=2)

    temp_data_manager.temp_files.append(models_file)
    return models_file


class TestModelDataValidation:
  """Test validation of model data structure."""

  def test_valid_openai_model(self, temp_data_manager):
    """Test valid OpenAI model configuration."""
    models_data = {
      "gpt-4": {
        "model": "gpt-4",
        "provider": "openai",
        "type": "chat",
        "max_tokens": 8192,
        "supports_functions": True
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("gpt-4")

      assert result["provider"] == "openai"
      assert result["type"] == "chat"
      assert result["max_tokens"] == 8192
      assert result["supports_functions"] is True

  def test_valid_anthropic_model(self, temp_data_manager):
    """Test valid Anthropic model configuration."""
    models_data = {
      "claude-3-opus-20240229": {
        "model": "claude-3-opus-20240229",
        "alias": "opus",
        "provider": "anthropic",
        "type": "chat",
        "max_tokens": 4096,
        "context_window": 200000
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("opus")

      assert result["provider"] == "anthropic"
      assert result["type"] == "chat"
      assert result["context_window"] == 200000

  def test_valid_embedding_model(self, temp_data_manager):
    """Test valid embedding model configuration."""
    models_data = {
      "text-embedding-3-large": {
        "model": "text-embedding-3-large",
        "provider": "openai",
        "type": "embedding",
        "dimensions": 3072,
        "max_input_tokens": 8191
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("text-embedding-3-large")

      assert result["type"] == "embedding"
      assert result["dimensions"] == 3072
      assert result["max_input_tokens"] == 8191

  def test_model_with_nested_config(self, temp_data_manager):
    """Test model with nested configuration objects."""
    models_data = {
      "custom-model": {
        "model": "custom-model",
        "provider": "custom",
        "config": {
          "api_endpoint": "https://api.custom.com/v1",
          "authentication": {
            "type": "bearer",
            "header": "Authorization"
          },
          "limits": {
            "requests_per_minute": 60,
            "tokens_per_minute": 40000
          }
        }
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file):
      result = get_canonical_model("custom-model")

      assert result["config"]["api_endpoint"] == "https://api.custom.com/v1"
      assert result["config"]["authentication"]["type"] == "bearer"
      assert result["config"]["limits"]["requests_per_minute"] == 60

  def _create_models_file(self, temp_data_manager, models_data):
    """Helper to create a temporary Models.json file."""
    temp_dir = temp_data_manager.create_temp_dir()
    models_file = os.path.join(temp_dir, "Models.json")

    with open(models_file, 'w') as f:
      json.dump(models_data, f, indent=2)

    temp_data_manager.temp_files.append(models_file)
    return models_file


class TestErrorHandling:
  """Test error handling scenarios."""

  def test_corrupted_models_file(self, temp_data_manager):
    """Test handling of corrupted Models.json file."""
    temp_dir = temp_data_manager.create_temp_dir()
    models_file = os.path.join(temp_dir, "Models.json")

    # Create file with corrupted JSON
    with open(models_file, 'w') as f:
      f.write('{"model1": {"name": "test"}, "model2": {invalid json}')

    with patch('models.model_manager.models_file', models_file), pytest.raises(json.JSONDecodeError):
      get_canonical_model("model1")

  def test_permission_denied_models_file(self):
    """Test handling of permission denied when reading Models.json."""
    with patch('builtins.open', side_effect=PermissionError("Access denied")), pytest.raises(PermissionError):
      get_canonical_model("gpt-4")

  def test_models_file_is_directory(self, temp_data_manager):
    """Test handling when Models.json path points to a directory."""
    temp_dir = temp_data_manager.create_temp_dir()
    models_path = os.path.join(temp_dir, "Models.json")
    os.makedirs(models_path)  # Create directory instead of file

    with patch('models.model_manager.models_file', models_path), pytest.raises((IsADirectoryError, PermissionError)):
      get_canonical_model("gpt-4")

  def test_none_model_name(self, temp_data_manager):
    """Test handling of None model name."""
    models_data = {"gpt-4": {"model": "gpt-4"}}
    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file), pytest.raises(ValueError, match="non-empty string"):
      get_canonical_model(None)

  def test_empty_model_name(self, temp_data_manager):
    """Test handling of empty model name."""
    models_data = {"gpt-4": {"model": "gpt-4"}}
    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file), pytest.raises(ValueError, match="non-empty string"):
      get_canonical_model("")

  def _create_models_file(self, temp_data_manager, models_data):
    """Helper to create a temporary Models.json file."""
    temp_dir = temp_data_manager.create_temp_dir()
    models_file = os.path.join(temp_dir, "Models.json")

    with open(models_file, 'w') as f:
      json.dump(models_data, f, indent=2)

    temp_data_manager.temp_files.append(models_file)
    return models_file


class TestLoggingIntegration:
  """Test logging integration in model manager."""

  def test_partial_match_warning_logged(self, temp_data_manager):
    """Test that partial match warning is logged."""
    models_data = {
      "gpt-4-turbo-preview": {
        "model": "gpt-4-turbo-preview",
        "provider": "openai"
      }
    }

    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file), patch('models.model_manager.logger') as mock_logger:
      get_canonical_model("gpt-4")  # Partial match

      # Should log warning about partial match
      mock_logger.warning.assert_called_once()
      warning_call = mock_logger.warning.call_args[0][0]
      assert "partial match" in warning_call.lower()
      assert "gpt-4-turbo-preview" in warning_call

  def test_model_not_found_error_logged(self, temp_data_manager):
    """Test that model not found error is logged."""
    models_data = {"gpt-4": {"model": "gpt-4"}}
    models_file = self._create_models_file(temp_data_manager, models_data)

    with patch('models.model_manager.models_file', models_file), patch('models.model_manager.logger') as mock_logger:
      with pytest.raises(KeyError):
        get_canonical_model("nonexistent-model")

      # Should log error about model not found
      mock_logger.error.assert_called_once()
      error_call = mock_logger.error.call_args[0][0]
      assert "not found" in error_call.lower()
      assert "nonexistent-model" in error_call

  def test_file_not_found_error_logged(self):
    """Test that file not found error is logged."""
    with patch('models.model_manager.models_file', "/nonexistent/Models.json"), \
         patch('models.model_manager.logger') as mock_logger, \
         pytest.raises(FileNotFoundError):
      get_canonical_model("gpt-4")

    # Should log error about file not found
    mock_logger.error.assert_called_once()
    error_call = mock_logger.error.call_args[0][0]
    assert "not found" in error_call.lower()

  def _create_models_file(self, temp_data_manager, models_data):
    """Helper to create a temporary Models.json file."""
    temp_dir = temp_data_manager.create_temp_dir()
    models_file = os.path.join(temp_dir, "Models.json")

    with open(models_file, 'w') as f:
      json.dump(models_data, f, indent=2)

    temp_data_manager.temp_files.append(models_file)
    return models_file

#fin
