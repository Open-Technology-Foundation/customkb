"""
Unit tests for prompt templates.
"""

import unittest
from unittest.mock import Mock, patch
from query.prompt_templates import (
    get_prompt_template, list_templates, validate_template_name,
    PROMPT_TEMPLATES
)


class TestPromptTemplates(unittest.TestCase):
  """Test cases for prompt template functionality."""
  
  def test_get_prompt_template_default(self):
    """Test getting the default template."""
    template = get_prompt_template('default')
    self.assertIn('system', template)
    self.assertIn('user', template)
    self.assertEqual(template['system'], 'You are a helpful assistant.')
    self.assertEqual(template['user'], '{reference_string}\n\n{query_text}')
    # Ensure description is removed
    self.assertNotIn('description', template)
  
  def test_get_prompt_template_instructive(self):
    """Test getting the instructive template."""
    template = get_prompt_template('instructive')
    self.assertIn('system', template)
    self.assertIn('user', template)
    self.assertIn('Based on the following reference materials:', template['user'])
    self.assertIn('Please answer this question:', template['user'])
    self.assertIn('{reference_string}', template['user'])
    self.assertIn('{query_text}', template['user'])
  
  def test_get_prompt_template_scholarly(self):
    """Test getting the scholarly template."""
    template = get_prompt_template('scholarly')
    self.assertIn('expert research assistant', template['system'])
    self.assertIn('Research Query:', template['user'])
    self.assertIn('comprehensive answer', template['user'])
  
  def test_get_prompt_template_with_custom_role(self):
    """Test getting a template with custom system role override."""
    custom_role = 'You are a specialized AI assistant for medical queries.'
    template = get_prompt_template('default', custom_role)
    self.assertEqual(template['system'], custom_role)
    # User template should remain unchanged
    self.assertEqual(template['user'], '{reference_string}\n\n{query_text}')
  
  def test_get_prompt_template_custom_role_not_default(self):
    """Test that default system role doesn't override template."""
    default_role = 'You are a helpful assistant.'
    template = get_prompt_template('scholarly', default_role)
    # Should keep the scholarly template's system role
    self.assertIn('expert research assistant', template['system'])
    self.assertNotEqual(template['system'], default_role)
  
  def test_get_prompt_template_invalid(self):
    """Test getting an invalid template raises error."""
    with self.assertRaises(ValueError) as context:
      get_prompt_template('nonexistent')
    self.assertIn('Unknown prompt template', str(context.exception))
    self.assertIn('Available templates:', str(context.exception))
  
  def test_list_templates(self):
    """Test listing all available templates."""
    templates = list_templates()
    self.assertIsInstance(templates, dict)
    # Check all expected templates are present
    expected_templates = ['default', 'instructive', 'scholarly', 'concise', 
                         'analytical', 'conversational', 'technical']
    for template in expected_templates:
      self.assertIn(template, templates)
      # Each should have a description
      self.assertIsInstance(templates[template], str)
      self.assertGreater(len(templates[template]), 0)
  
  def test_validate_template_name(self):
    """Test template name validation."""
    # Valid templates
    self.assertTrue(validate_template_name('default'))
    self.assertTrue(validate_template_name('instructive'))
    self.assertTrue(validate_template_name('scholarly'))
    
    # Invalid templates
    self.assertFalse(validate_template_name('invalid'))
    self.assertFalse(validate_template_name(''))
    self.assertFalse(validate_template_name('DEFAULT'))  # Case sensitive
  
  def test_template_formatting(self):
    """Test that templates can be properly formatted."""
    template = get_prompt_template('instructive')
    
    # Test formatting with sample data
    reference = "Sample reference content"
    query = "What is the meaning of life?"
    
    formatted_user = template['user'].format(
      reference_string=reference,
      query_text=query
    )
    
    self.assertIn(reference, formatted_user)
    self.assertIn(query, formatted_user)
    self.assertNotIn('{reference_string}', formatted_user)
    self.assertNotIn('{query_text}', formatted_user)
  
  def test_all_templates_have_required_fields(self):
    """Test that all templates have the required fields."""
    for template_name in PROMPT_TEMPLATES:
      template_data = PROMPT_TEMPLATES[template_name]
      # Check required fields exist
      self.assertIn('system', template_data, f"Template {template_name} missing 'system'")
      self.assertIn('user', template_data, f"Template {template_name} missing 'user'")
      self.assertIn('description', template_data, f"Template {template_name} missing 'description'")
      
      # Check placeholders are present in user template
      self.assertIn('{reference_string}', template_data['user'], 
                   f"Template {template_name} missing reference_string placeholder")
      self.assertIn('{query_text}', template_data['user'],
                   f"Template {template_name} missing query_text placeholder")
  
  def test_template_copy_isolation(self):
    """Test that getting a template returns a copy, not the original."""
    template1 = get_prompt_template('default')
    template2 = get_prompt_template('default')
    
    # Modify template1
    template1['system'] = 'Modified system'
    
    # template2 should not be affected
    self.assertEqual(template2['system'], 'You are a helpful assistant.')
    
    # Original should also not be affected
    template3 = get_prompt_template('default')
    self.assertEqual(template3['system'], 'You are a helpful assistant.')


if __name__ == '__main__':
  unittest.main()

#fin