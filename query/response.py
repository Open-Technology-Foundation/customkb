#!/usr/bin/env python
"""
AI response generation for CustomKB queries.

This module handles AI model clients, response generation,
prompt templates, and model-specific formatting.
"""

import os
import sys
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic

from utils.logging_config import get_logger
from utils.security_utils import validate_api_key
from utils.exceptions import APIError, AuthenticationError, ModelError
from models.model_manager import get_canonical_model

logger = get_logger(__name__)

# Google AI client
try:
  from google import genai
  GOOGLE_AI_AVAILABLE = True
except ImportError:
  GOOGLE_AI_AVAILABLE = False

# Global clients
openai_client = None
async_openai_client = None
anthropic_client = None
async_anthropic_client = None
google_client = None
xai_client = None
async_xai_client = None
llama_client = None


def load_and_validate_api_keys():
  """Load and validate API keys securely."""
  keys = {}
  
  # Load OpenAI API key
  openai_key = os.getenv('OPENAI_API_KEY')
  if openai_key and validate_api_key(openai_key, 'sk-', 40):
    keys['openai'] = openai_key
  
  # Load Anthropic API key
  anthropic_key = os.getenv('ANTHROPIC_API_KEY')
  if anthropic_key and validate_api_key(anthropic_key, 'sk-ant-', 95):
    keys['anthropic'] = anthropic_key
  
  # Load xAI API key (optional)
  xai_key = os.getenv('XAI_API_KEY')
  if xai_key and validate_api_key(xai_key, 'xai-', 20):
    keys['xai'] = xai_key
  
  # Load Google API key (optional)
  google_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
  if google_key and validate_api_key(google_key, min_length=20):
    keys['google'] = google_key
  
  return keys


def initialize_clients():
  """Initialize AI clients with available API keys."""
  global openai_client, async_openai_client, anthropic_client, async_anthropic_client, google_client, xai_client, async_xai_client, llama_client
  
  try:
    keys = load_and_validate_api_keys()
    
    # Initialize OpenAI clients
    if 'openai' in keys:
      openai_client = OpenAI(api_key=keys['openai'], timeout=300.0)  # 5 minute timeout
      async_openai_client = AsyncOpenAI(api_key=keys['openai'], timeout=300.0)  # 5 minute timeout
      logger.debug("OpenAI clients initialized")
    
    # Initialize Anthropic clients
    if 'anthropic' in keys:
      anthropic_client = Anthropic(api_key=keys['anthropic'])
      async_anthropic_client = AsyncAnthropic(api_key=keys['anthropic'])
      logger.debug("Anthropic clients initialized")
    
    # Initialize xAI clients
    if 'xai' in keys:
      xai_client = OpenAI(api_key=keys['xai'], base_url="https://api.x.ai/v1")
      async_xai_client = AsyncOpenAI(api_key=keys['xai'], base_url="https://api.x.ai/v1")
      logger.debug("xAI clients initialized")
    
    # Initialize Google AI client
    if 'google' in keys and GOOGLE_AI_AVAILABLE:
      google_client = genai.Client(api_key=keys['google'])
      logger.debug("Google AI client initialized")
    
    # Initialize Llama client (local)
    try:
      llama_client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
      logger.debug("Llama client initialized")
    except Exception as e:
      logger.debug(f"Llama client initialization failed: {e}")
    
  except Exception as e:
    logger.error(f"Client initialization failed: {e}")
    raise AuthenticationError(f"Failed to initialize AI clients: {e}") from e


# Initialize clients on module import
initialize_clients()


def _is_reasoning_model(model: str) -> bool:
  """
  Check if a model supports reasoning parameter.
  
  Args:
      model: Model name (may include date suffix)
      
  Returns:
      True if model supports reasoning
  """
  # OpenAI o1 models support reasoning
  reasoning_models = ['o1-preview', 'o1-mini']
  
  # Check base model name
  base_model = model.split('-')[0] + '-' + model.split('-')[1] if '-' in model else model
  return base_model in reasoning_models


def _is_gpt5_model(model: str) -> bool:
  """
  Check if model is GPT-5 series (which doesn't support temperature).
  
  Args:
      model: Model name
      
  Returns:
      True if GPT-5 series model
  """
  model_lower = model.lower()
  return model_lower.startswith('gpt-5') or model_lower.startswith('gpt5')


def format_messages_for_responses_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """
  Format messages for OpenAI Responses API.
  
  Args:
      messages: List of message dictionaries
      
  Returns:
      Formatted messages for Responses API
  """
  formatted_messages = []
  
  for message in messages:
    role = message.get('role')
    content = message.get('content')
    
    if role == 'system':
      # System messages go to developer role
      formatted_messages.append({
        'role': 'developer',
        'content': content
      })
    elif role in ['user', 'assistant']:
      formatted_messages.append({
        'role': role,
        'content': content
      })
  
  return formatted_messages


def _extract_content_from_response(data: Dict[str, Any]) -> str:
  """
  Extract content from API response data.
  
  Args:
      data: Response data from API
      
  Returns:
      Extracted content string
  """
  try:
    # OpenAI format
    if 'choices' in data:
      if data['choices'] and 'message' in data['choices'][0]:
        return data['choices'][0]['message'].get('content', '')
      elif data['choices'] and 'text' in data['choices'][0]:
        return data['choices'][0]['text']
    
    # Anthropic format
    if 'content' in data:
      if isinstance(data['content'], list) and data['content']:
        return data['content'][0].get('text', '')
      elif isinstance(data['content'], str):
        return data['content']
    
    # Direct content
    if isinstance(data, str):
      return data
    
    return str(data)
    
  except Exception as e:
    logger.error(f"Failed to extract content from response: {e}")
    return ""


def get_prompt_template(template_name: str = None) -> Dict[str, str]:
  """
  Get prompt template by name.
  
  Args:
      template_name: Name of the template to use
      
  Returns:
      Dictionary with system and user prompt templates
  """
  templates = {
    'default': {
      'system': "You are a helpful AI assistant. Answer questions based on the provided context.",
      'user': "Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a helpful answer based on the context above."
    },
    'instructive': {
      'system': "You are an AI assistant that answers questions based strictly on provided reference materials.",
      'user': """Based on the following reference materials:
{context}

Please answer this question: {query}

Instructions:
- Base your answer solely on the provided references
- If the references don't contain relevant information, state this clearly
- Be concise but thorough in your response
- Cite specific sources when possible"""
    },
    'scholarly': {
      'system': "You are a scholarly AI assistant that provides comprehensive, well-researched answers with proper citations.",
      'user': """Reference Materials:
{context}

Research Question: {query}

Please provide a scholarly analysis that:
1. Synthesizes information from the references
2. Cites specific sources
3. Acknowledges limitations in the available information
4. Provides a comprehensive yet focused response"""
    },
    'concise': {
      'system': "You are an AI assistant that provides brief, direct answers.",
      'user': "Context: {context}\n\nQuestion: {query}\n\nProvide a concise, direct answer:"
    },
    'analytical': {
      'system': "You are an analytical AI assistant that breaks down complex information systematically.",
      'user': """Available Information:
{context}

Analysis Request: {query}

Please provide a structured analysis:
1. Key Information: Identify relevant facts from the references
2. Analysis: Examine the information in relation to the query  
3. Synthesis: Provide a clear, evidence-based response
4. Limitations: Note any gaps in the available information"""
    },
    'conversational': {
      'system': "You are a friendly, conversational AI assistant that maintains accuracy while being approachable.",
      'user': """Hi! I have some information that might help answer your question:

{context}

Your question: {query}

Let me help you with that based on what I found!"""
    },
    'technical': {
      'system': "You are a technical AI assistant with expertise in providing detailed, precise answers for expert users.",
      'user': """Technical Documentation:
{context}

Technical Query: {query}

Please provide a detailed technical response that:
- Uses precise terminology
- Includes relevant technical details
- Maintains accuracy and depth
- Addresses implementation considerations where applicable"""
    }
  }
  
  return templates.get(template_name, templates['default'])


async def generate_openai_response(messages: List[Dict[str, Any]], model: str, 
                                  temperature: float = 0.7, max_tokens: int = 2000) -> str:
  """
  Generate response using OpenAI models.
  
  Args:
      messages: List of message dictionaries
      model: OpenAI model name
      temperature: Response randomness
      max_tokens: Maximum response length
      
  Returns:
      Generated response text
  """
  if not async_openai_client:
    raise APIError("OpenAI client not initialized")
  
  try:
    # Check if this is a reasoning model
    if _is_reasoning_model(model):
      # Use reasoning parameter for o1 models
      response = await async_openai_client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning=True,
        max_completion_tokens=max_tokens
      )
    elif _is_gpt5_model(model):
      # GPT-5 models require reasoning_effort or verbosity parameter
      # They don't support temperature parameter
      # Log message size for debugging
      total_chars = sum(len(msg.get('content', '')) for msg in messages)
      logger.info(f"Sending {total_chars} characters to GPT-5 model {model}")
      
      # Warn if context is very large
      if total_chars > 100000:  # ~100KB
        logger.warning(f"Context size ({total_chars} chars) may exceed model limits. Consider reducing --top-k or --context-scope")
      
      try:
        # GPT-5 requires reasoning_effort parameter
        # Use 'minimal' for faster responses (medium can timeout)
        response = await async_openai_client.chat.completions.create(
          model=model,
          messages=messages,
          max_completion_tokens=max_tokens,
          reasoning_effort='minimal',  # Required for GPT-5 - minimal is faster
          verbosity='low'  # Low verbosity for faster responses
        )
        # Debug logging
        logger.info(f"GPT-5 response received: finish_reason={response.choices[0].finish_reason if response.choices else 'no choices'}")
        if response.choices and response.choices[0].message:
          logger.info(f"GPT-5 content length: {len(response.choices[0].message.content or '')}")
      except Exception as api_error:
        logger.error(f"GPT-5 API call failed: {api_error}")
        # Try to extract more details
        if hasattr(api_error, 'status_code'):
          logger.error(f"Status code: {api_error.status_code}")
        if hasattr(api_error, 'body'):
          logger.error(f"Response body: {api_error.body}")
        raise
    else:
      # Other models (GPT-4, GPT-3.5, etc) - ALL use max_completion_tokens now
      response = await async_openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens
      )
    
    # Check if we got a valid response
    if not response or not response.choices:
      logger.error(f"Empty or invalid response from OpenAI for model {model}")
      return None
    
    return response.choices[0].message.content
    
  except Exception as e:
    logger.error(f"OpenAI API error: {e}")
    # Provide more detail about the error
    error_msg = str(e)
    if hasattr(e, 'response'):
      try:
        error_detail = e.response.json()
        error_msg = f"{error_msg} - Details: {error_detail}"
      except:
        pass
    raise APIError(f"OpenAI request failed for model {model}: {error_msg}") from e


async def generate_anthropic_response(messages: List[Dict[str, Any]], model: str,
                                     temperature: float = 0.7, max_tokens: int = 2000) -> str:
  """
  Generate response using Anthropic Claude models.
  
  Args:
      messages: List of message dictionaries
      model: Anthropic model name
      temperature: Response randomness
      max_tokens: Maximum response length
      
  Returns:
      Generated response text
  """
  if not async_anthropic_client:
    raise APIError("Anthropic client not initialized")
  
  try:
    # Extract system message
    system_message = ""
    user_messages = []
    
    for msg in messages:
      if msg['role'] == 'system':
        system_message = msg['content']
      else:
        user_messages.append(msg)
    
    # Create Anthropic request
    request_params = {
      'model': model,
      'messages': user_messages,
      'temperature': temperature,
      'max_tokens': max_tokens
    }
    
    if system_message:
      request_params['system'] = system_message
    
    response = await async_anthropic_client.messages.create(**request_params)
    
    return response.content[0].text
    
  except Exception as e:
    logger.error(f"Anthropic API error: {e}")
    raise APIError(f"Anthropic request failed: {e}") from e


async def generate_google_response(messages: List[Dict[str, Any]], model: str,
                                  temperature: float = 0.7, max_tokens: int = 2000) -> str:
  """
  Generate response using Google AI models.
  
  Args:
      messages: List of message dictionaries
      model: Google model name
      temperature: Response randomness
      max_tokens: Maximum response length
      
  Returns:
      Generated response text
  """
  if not google_client:
    raise APIError("Google AI client not initialized")
  
  try:
    # Convert messages to Google format
    prompt_parts = []
    for msg in messages:
      role = msg['role']
      content = msg['content']
      
      if role == 'system':
        prompt_parts.append(f"System: {content}")
      elif role == 'user':
        prompt_parts.append(f"User: {content}")
      elif role == 'assistant':
        prompt_parts.append(f"Assistant: {content}")
    
    prompt = "\n\n".join(prompt_parts)
    
    # Generate response
    response = google_client.models.generate_content(
      model=f"models/{model}",
      contents=prompt,
      generation_config={
        'temperature': temperature,
        'max_output_tokens': max_tokens
      }
    )
    
    return response.text
    
  except Exception as e:
    logger.error(f"Google AI API error: {e}")
    raise APIError(f"Google AI request failed: {e}") from e


async def generate_xai_response(messages: List[Dict[str, Any]], model: str,
                               temperature: float = 0.7, max_tokens: int = 2000) -> str:
  """
  Generate response using xAI models.
  
  Args:
      messages: List of message dictionaries
      model: xAI model name
      temperature: Response randomness
      max_tokens: Maximum response length
      
  Returns:
      Generated response text
  """
  if not async_xai_client:
    raise APIError("xAI client not initialized")
  
  try:
    response = await async_xai_client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens
    )
    
    return response.choices[0].message.content
    
  except Exception as e:
    logger.error(f"xAI API error: {e}")
    raise APIError(f"xAI request failed: {e}") from e


async def generate_llama_response(messages: List[Dict[str, Any]], model: str,
                                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
  """
  Generate response using local Llama models via Ollama.
  
  Args:
      messages: List of message dictionaries
      model: Llama model name
      temperature: Response randomness
      max_tokens: Maximum response length
      
  Returns:
      Generated response text
  """
  if not llama_client:
    raise APIError("Llama client not initialized")
  
  try:
    # Create async client for Llama
    async_llama_client = AsyncOpenAI(
      api_key='ollama', 
      base_url='http://localhost:11434/v1'
    )
    
    response = await async_llama_client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens
    )
    
    return response.choices[0].message.content
    
  except Exception as e:
    logger.error(f"Llama API error: {e}")
    raise APIError(f"Llama request failed: {e}") from e


async def generate_ai_response(kb: Any, reference_string: str, query_text: str,
                              prompt_template: str = None) -> str:
  """
  Generate AI response using the configured model.
  
  Args:
      kb: KnowledgeBase instance
      reference_string: Context from search results
      query_text: Original query text
      prompt_template: Optional prompt template name
      
  Returns:
      Generated response text
  """
  model_name = kb.query_model  # Default to the input model name
  try:
    # Get model configuration
    model_info = get_canonical_model(kb.query_model)
    model_name = model_info['model']
    provider = model_info.get('provider', 'openai')
    
    # Get prompt template
    template_name = prompt_template or getattr(kb, 'query_prompt_template', 'default')
    template = get_prompt_template(template_name)
    
    # Format messages
    system_prompt = template['system']
    
    # Override system role if specified
    if hasattr(kb, 'query_role') and kb.query_role:
      system_prompt = kb.query_role
    
    user_prompt = template['user'].format(
      context=reference_string,
      query=query_text
    )
    
    messages = [
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': user_prompt}
    ]
    
    # Get generation parameters
    temperature = getattr(kb, 'query_temperature', 0.7)
    max_tokens = getattr(kb, 'query_max_tokens', 2000)
    
    logger.info(f"Generating response with {provider} model: {model_name}")
    
    # Route to appropriate provider
    if provider == 'anthropic':
      response = await generate_anthropic_response(messages, model_name, temperature, max_tokens)
    elif provider == 'google':
      response = await generate_google_response(messages, model_name, temperature, max_tokens)
    elif provider == 'xai':
      response = await generate_xai_response(messages, model_name, temperature, max_tokens)
    elif provider == 'local' or 'llama' in model_name.lower():
      response = await generate_llama_response(messages, model_name, temperature, max_tokens)
    else:
      # Default to OpenAI
      response = await generate_openai_response(messages, model_name, temperature, max_tokens)
    
    if not response:
      raise ModelError(model_name, "Empty response from AI model")
    
    logger.info(f"Generated response: {len(response)} characters")
    return response
    
  except ModelError:
    # Re-raise ModelError as-is
    raise
  except Exception as e:
    logger.error(f"AI response generation failed: {e}")
    raise ModelError(model_name, f"Failed to generate AI response: {e}") from e


#fin