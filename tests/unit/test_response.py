#!/usr/bin/env python3
"""
Unit tests for query response generation functionality.

Tests LLM response generation, prompt formatting, and API interactions.
"""


import pytest

from query.response import (
    _extract_content_from_response,
    _is_gpt5_model,
    _is_reasoning_model,
    format_messages_for_responses_api,
    get_prompt_template,
)


class TestIsReasoningModel:
    """Test reasoning model detection."""

    def test_o1_preview_is_reasoning(self):
        """Test that o1-preview is detected as reasoning model."""
        assert _is_reasoning_model("o1-preview") is True

    def test_o1_mini_is_reasoning(self):
        """Test that o1-mini is detected as reasoning model."""
        assert _is_reasoning_model("o1-mini") is True

    def test_o1_preview_with_date_suffix(self):
        """Test o1-preview with date suffix."""
        assert _is_reasoning_model("o1-preview-2024-09-12") is True

    def test_o1_mini_with_date_suffix(self):
        """Test o1-mini with date suffix."""
        assert _is_reasoning_model("o1-mini-2024-09-12") is True

    def test_gpt4_not_reasoning(self):
        """Test that GPT-4 models are not reasoning models."""
        assert _is_reasoning_model("gpt-4") is False
        assert _is_reasoning_model("gpt-4-turbo") is False
        assert _is_reasoning_model("gpt-4o") is False

    def test_gpt35_not_reasoning(self):
        """Test that GPT-3.5 is not a reasoning model."""
        assert _is_reasoning_model("gpt-3.5-turbo") is False

    def test_claude_not_reasoning(self):
        """Test that Claude models are not reasoning models."""
        assert _is_reasoning_model("claude-3-opus") is False
        assert _is_reasoning_model("claude-3-sonnet") is False

    def test_empty_string_not_reasoning(self):
        """Test empty string is not detected as reasoning model."""
        assert _is_reasoning_model("") is False

    def test_o1_prefix_not_enough(self):
        """Test that just starting with o1 isn't enough."""
        assert _is_reasoning_model("o1-custom-model") is False


class TestIsGPT5Model:
    """Test GPT-5 model detection."""

    def test_gpt5_lowercase(self):
        """Test GPT-5 detection with lowercase."""
        assert _is_gpt5_model("gpt-5") is True

    def test_gpt5_uppercase(self):
        """Test GPT-5 detection with uppercase."""
        assert _is_gpt5_model("GPT-5") is True

    def test_gpt5_mixed_case(self):
        """Test GPT-5 detection with mixed case."""
        assert _is_gpt5_model("GpT-5") is True

    def test_gpt5_no_dash(self):
        """Test GPT-5 without dash."""
        assert _is_gpt5_model("gpt5") is True

    def test_gpt5_with_suffix(self):
        """Test GPT-5 with suffix."""
        assert _is_gpt5_model("gpt-5-turbo") is True
        assert _is_gpt5_model("gpt5-preview") is True

    def test_gpt4_not_gpt5(self):
        """Test that GPT-4 models are not GPT-5."""
        assert _is_gpt5_model("gpt-4") is False
        assert _is_gpt5_model("gpt-4o") is False
        assert _is_gpt5_model("gpt-4-turbo") is False

    def test_gpt35_not_gpt5(self):
        """Test that GPT-3.5 is not GPT-5."""
        assert _is_gpt5_model("gpt-3.5-turbo") is False

    def test_non_gpt_model_not_gpt5(self):
        """Test that non-GPT models are not GPT-5."""
        assert _is_gpt5_model("claude-3-opus") is False
        assert _is_gpt5_model("gemini-pro") is False

    def test_empty_string_not_gpt5(self):
        """Test empty string is not GPT-5."""
        assert _is_gpt5_model("") is False


class TestFormatMessagesForResponsesAPI:
    """Test message formatting for Responses API."""

    def test_format_system_message(self):
        """Test that system messages are converted to developer role."""
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}
        ]

        result = format_messages_for_responses_api(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'developer'
        assert result[0]['content'] == 'You are a helpful assistant.'

    def test_format_user_message(self):
        """Test that user messages are preserved."""
        messages = [
            {'role': 'user', 'content': 'What is AI?'}
        ]

        result = format_messages_for_responses_api(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'What is AI?'

    def test_format_assistant_message(self):
        """Test that assistant messages are preserved."""
        messages = [
            {'role': 'assistant', 'content': 'AI stands for Artificial Intelligence.'}
        ]

        result = format_messages_for_responses_api(messages)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'AI stands for Artificial Intelligence.'

    def test_format_conversation(self):
        """Test formatting a full conversation."""
        messages = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'Tell me about AI'}
        ]

        result = format_messages_for_responses_api(messages)

        assert len(result) == 4
        assert result[0]['role'] == 'developer'
        assert result[1]['role'] == 'user'
        assert result[2]['role'] == 'assistant'
        assert result[3]['role'] == 'user'

    def test_format_empty_messages(self):
        """Test formatting empty message list."""
        result = format_messages_for_responses_api([])

        assert result == []

    def test_format_preserves_content(self):
        """Test that content is preserved exactly."""
        messages = [
            {'role': 'user', 'content': 'Complex\n\nMultiline\nContent\n\n\twith\ttabs'}
        ]

        result = format_messages_for_responses_api(messages)

        assert result[0]['content'] == 'Complex\n\nMultiline\nContent\n\n\twith\ttabs'

    def test_format_unknown_role_ignored(self):
        """Test that unknown roles are ignored."""
        messages = [
            {'role': 'unknown', 'content': 'Test'},
            {'role': 'user', 'content': 'Hello'}
        ]

        result = format_messages_for_responses_api(messages)

        # Should only include the user message
        assert len(result) == 1
        assert result[0]['role'] == 'user'


class TestExtractContentFromResponse:
    """Test content extraction from API responses."""

    def test_extract_openai_format(self):
        """Test extracting content from OpenAI format."""
        data = {
            'choices': [
                {'message': {'content': 'This is the response'}}
            ]
        }

        result = _extract_content_from_response(data)

        assert result == 'This is the response'

    def test_extract_openai_text_format(self):
        """Test extracting from OpenAI text format."""
        data = {
            'choices': [
                {'text': 'This is text response'}
            ]
        }

        result = _extract_content_from_response(data)

        assert result == 'This is text response'

    def test_extract_anthropic_format_list(self):
        """Test extracting from Anthropic format (list)."""
        data = {
            'content': [
                {'text': 'Anthropic response'}
            ]
        }

        result = _extract_content_from_response(data)

        assert result == 'Anthropic response'

    def test_extract_anthropic_format_string(self):
        """Test extracting from Anthropic format (string)."""
        data = {
            'content': 'Direct content string'
        }

        result = _extract_content_from_response(data)

        assert result == 'Direct content string'

    def test_extract_direct_string(self):
        """Test extracting when data is already a string."""
        data = 'Direct string response'

        result = _extract_content_from_response(data)

        assert result == 'Direct string response'

    def test_extract_empty_choices(self):
        """Test extracting with empty choices list."""
        data = {'choices': []}

        result = _extract_content_from_response(data)

        # Should return empty string or fallback
        assert isinstance(result, str)

    def test_extract_missing_content(self):
        """Test extracting when content is missing."""
        data = {
            'choices': [
                {'message': {}}
            ]
        }

        result = _extract_content_from_response(data)

        assert result == ''

    def test_extract_empty_anthropic_content(self):
        """Test extracting from empty Anthropic content."""
        data = {'content': []}

        result = _extract_content_from_response(data)

        # Should handle gracefully
        assert isinstance(result, str)

    def test_extract_handles_none(self):
        """Test that None values are handled."""
        data = None

        result = _extract_content_from_response(data)

        # Should convert to string
        assert isinstance(result, str)

    def test_extract_complex_nested_structure(self):
        """Test extraction from complex nested structure."""
        data = {
            'choices': [
                {
                    'message': {
                        'content': 'Response with\nmultiple\nlines'
                    }
                }
            ]
        }

        result = _extract_content_from_response(data)

        assert result == 'Response with\nmultiple\nlines'

    def test_extract_preserves_formatting(self):
        """Test that formatting is preserved."""
        data = {
            'choices': [
                {'message': {'content': '  Spaces  \n\tTabs\t  '}}
            ]
        }

        result = _extract_content_from_response(data)

        assert result == '  Spaces  \n\tTabs\t  '


class TestGetPromptTemplate:
    """Test prompt template retrieval."""

    def test_get_default_template(self):
        """Test getting default template."""
        template = get_prompt_template()

        assert 'system' in template
        assert 'user' in template
        assert isinstance(template['system'], str)
        assert isinstance(template['user'], str)

    def test_get_default_template_explicit(self):
        """Test getting default template explicitly."""
        template = get_prompt_template('default')

        assert 'system' in template
        assert 'user' in template

    def test_get_instructive_template(self):
        """Test getting instructive template."""
        template = get_prompt_template('instructive')

        assert 'system' in template
        assert 'user' in template
        assert 'reference materials' in template['user'].lower()

    def test_get_scholarly_template(self):
        """Test getting scholarly template."""
        template = get_prompt_template('scholarly')

        assert 'system' in template
        assert 'user' in template
        assert 'scholarly' in template['system'].lower()

    def test_get_concise_template(self):
        """Test getting concise template."""
        template = get_prompt_template('concise')

        assert 'system' in template
        assert 'user' in template
        assert 'concise' in template['system'].lower() or 'concise' in template['user'].lower()

    def test_get_analytical_template(self):
        """Test getting analytical template."""
        template = get_prompt_template('analytical')

        assert 'system' in template
        assert 'user' in template
        assert 'analytical' in template['system'].lower()

    def test_get_conversational_template(self):
        """Test getting conversational template."""
        template = get_prompt_template('conversational')

        assert 'system' in template
        assert 'user' in template
        assert 'conversational' in template['system'].lower() or 'friendly' in template['system'].lower()

    def test_get_technical_template(self):
        """Test getting technical template."""
        template = get_prompt_template('technical')

        assert 'system' in template
        assert 'user' in template
        assert 'technical' in template['system'].lower()

    def test_get_unknown_template_returns_default(self):
        """Test that unknown template name returns default."""
        template = get_prompt_template('nonexistent')

        # Should return default template
        default_template = get_prompt_template('default')
        assert template == default_template

    def test_template_has_placeholders(self):
        """Test that templates contain required placeholders."""
        templates = ['default', 'instructive', 'scholarly', 'concise', 'analytical', 'conversational', 'technical']

        for template_name in templates:
            template = get_prompt_template(template_name)

            # User prompt should have context and query placeholders
            assert '{context}' in template['user']
            assert '{query}' in template['user']

    def test_all_templates_are_strings(self):
        """Test that all template values are strings."""
        templates = ['default', 'instructive', 'scholarly', 'concise', 'analytical', 'conversational', 'technical']

        for template_name in templates:
            template = get_prompt_template(template_name)

            assert isinstance(template['system'], str)
            assert isinstance(template['user'], str)

    def test_templates_are_not_empty(self):
        """Test that all templates have non-empty content."""
        templates = ['default', 'instructive', 'scholarly', 'concise', 'analytical', 'conversational', 'technical']

        for template_name in templates:
            template = get_prompt_template(template_name)

            assert len(template['system']) > 0
            assert len(template['user']) > 0


class TestEdgeCases:
    """Test edge cases for response functions."""

    def test_is_reasoning_model_with_special_chars(self):
        """Test reasoning model detection with special characters."""
        # Special characters should not match - model names must be exact
        assert _is_reasoning_model("o1-preview!@#") is False
        # But valid date suffixes should work
        assert _is_reasoning_model("o1-preview-2024-09-12") is True

    def test_is_gpt5_with_whitespace(self):
        """Test GPT-5 detection doesn't match whitespace."""
        assert _is_gpt5_model(" gpt-5") is False  # Leading space
        assert _is_gpt5_model("gpt-5 ") is True   # Trailing space OK due to startswith

    def test_format_messages_missing_role(self):
        """Test formatting messages with missing role."""
        messages = [
            {'content': 'Test content'}
        ]

        result = format_messages_for_responses_api(messages)

        # Should handle gracefully
        assert isinstance(result, list)

    def test_format_messages_missing_content(self):
        """Test formatting messages with missing content."""
        messages = [
            {'role': 'user'}
        ]

        result = format_messages_for_responses_api(messages)

        # Should include message even without content
        assert len(result) == 1

    def test_extract_content_from_malformed_data(self):
        """Test extracting content from malformed data."""
        data = {'unexpected': 'structure'}

        result = _extract_content_from_response(data)

        # Should return string representation or empty
        assert isinstance(result, str)

    def test_template_none_input(self):
        """Test get_prompt_template with None input."""
        template = get_prompt_template(None)

        # Should return default template
        assert 'system' in template
        assert 'user' in template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


#fin
