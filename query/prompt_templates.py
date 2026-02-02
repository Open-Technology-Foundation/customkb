"""
Prompt templates for different query styles and use cases.

This module provides various prompt templates that can be used to structure
how queries are presented to LLMs, improving response quality and consistency.
"""




PROMPT_TEMPLATES = {
  'default': {
    'system': 'You are a helpful assistant.',
    'user': '{reference_string}\n\n{query_text}',
    'description': 'Simple format matching original behavior - minimal instructions'
  },

  'instructive': {
    'system': 'You are a helpful assistant with access to reference materials. Provide accurate, well-structured answers based on the provided context.',
    'user': '''Based on the following reference materials:

{reference_string}

Please answer this question: {query_text}

Instructions:
- Base your answer solely on the provided references
- If the references don't contain relevant information, state this clearly
- Be concise but thorough in your response''',
    'description': 'Clear instructions for context-based answering with explicit guidelines'
  },

  'scholarly': {
    'system': 'You are an expert research assistant providing detailed, accurate answers based on provided sources. Always maintain academic rigor and cite your sources.',
    'user': '''Reference Materials:
{reference_string}

Research Query: {query_text}

Please provide a comprehensive answer following these guidelines:
1. Base your response exclusively on the provided references
2. Cite specific sections or sources when making claims
3. Maintain an objective, analytical tone
4. If the references are insufficient, explicitly state what information is missing
5. Structure your response with clear sections if appropriate''',
    'description': 'Academic style with emphasis on citations and comprehensive analysis'
  },

  'concise': {
    'system': 'You are a concise assistant. Provide brief, direct answers based on the provided context. Avoid unnecessary elaboration.',
    'user': '''Context:
{reference_string}

Question: {query_text}

Answer briefly based only on the context provided. If the answer is not in the context, say so directly.''',
    'description': 'Minimal, direct responses for quick answers'
  },

  'analytical': {
    'system': 'You are an analytical assistant that provides structured, evidence-based responses. Break down complex topics and analyze them systematically.',
    'user': '''Available Information:
{reference_string}

Analysis Request: {query_text}

Please provide a structured analysis:
1. Key Information: Identify relevant facts from the references
2. Analysis: Examine the information in relation to the query
3. Synthesis: Provide a clear, evidence-based response
4. Limitations: Note any gaps in the available information

Use bullet points or numbered lists where appropriate for clarity.''',
    'description': 'Structured analytical approach with systematic breakdown'
  },

  'conversational': {
    'system': 'You are a friendly, conversational assistant. Provide helpful answers in a natural, approachable tone while staying accurate to the provided information.',
    'user': '''Here's what I found in the knowledgebase:

{reference_string}

Your question: {query_text}

Let me help you with that based on the information available. I'll keep my response friendly and easy to understand.''',
    'description': 'Friendly, conversational tone while maintaining accuracy'
  },

  'technical': {
    'system': 'You are a technical expert assistant. Provide precise, detailed technical answers with appropriate terminology and depth.',
    'user': '''Technical Documentation:
{reference_string}

Technical Query: {query_text}

Provide a detailed technical response:
- Use precise technical terminology
- Include relevant details and specifications
- Explain complex concepts clearly
- Reference specific documentation sections
- If information is incomplete, specify what additional details would be needed''',
    'description': 'Technical depth with precise terminology for expert users'
  }
}


def get_prompt_template(template_name: str, custom_system_role: str | None = None) -> dict[str, str]:
  """
  Get a prompt template by name, with optional custom system role override.

  Args:
      template_name: Name of the template to retrieve
      custom_system_role: Optional custom system role to override template default

  Returns:
      Dictionary with 'system' and 'user' keys containing the prompt templates

  Raises:
      ValueError: If template_name is not found
  """
  if template_name not in PROMPT_TEMPLATES:
    available = ', '.join(PROMPT_TEMPLATES.keys())
    raise ValueError(f"Unknown prompt template '{template_name}'. Available templates: {available}")

  template = PROMPT_TEMPLATES[template_name].copy()

  # Remove description from returned template
  template.pop('description', None)

  # Override system role if custom one provided and not using default
  if custom_system_role and custom_system_role != 'You are a helpful assistant.':
    template['system'] = custom_system_role

  return template


def list_templates() -> dict[str, str]:
  """
  Get a list of all available templates with their descriptions.

  Returns:
      Dictionary mapping template names to their descriptions
  """
  return {
    name: template.get('description', 'No description available')
    for name, template in PROMPT_TEMPLATES.items()
  }


def validate_template_name(template_name: str) -> bool:
  """
  Check if a template name is valid.

  Args:
      template_name: Name of the template to validate

  Returns:
      True if template exists, False otherwise
  """
  return template_name in PROMPT_TEMPLATES


#fin
