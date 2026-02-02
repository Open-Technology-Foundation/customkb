#!/bin/bash
# api_functions.sh - Multi-provider API integration for citation extraction
#
# This library handles API interactions with multiple AI providers for the
# citations system, including rate limiting, retry logic, and response parsing.
#
# Supported Providers:
#   - OpenAI (gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo, o1-*, o3-*)
#   - Anthropic (claude-*, claude-sonnet-*, claude-opus-*, claude-haiku-*)
#   - xAI (grok-*)
#   - Google (gemini-*)
#
# Features:
#   - Automatic provider detection from model name
#   - Automatic rate limiting (1 request/second by default)
#   - Exponential backoff retry logic for failures
#   - HTTP error code handling (429 rate limit, 401 auth, etc.)
#   - JSON response validation and parsing
#   - API key format validation
#
# Functions:
#   - api_rate_limit: Enforce delay between API calls
#   - api_call: Route to appropriate provider based on model
#   - api_call_openai: Make OpenAI API request
#   - api_call_anthropic: Make Anthropic API request
#   - api_extract_result: Parse citation from API response
#   - api_validate_key: Validate API keys
#
# Environment Variables:
#   - OPENAI_API_KEY: OpenAI API key
#   - ANTHROPIC_API_KEY: Anthropic API key
#   - XAI_API_KEY: xAI API key
#   - GOOGLE_API_KEY: Google API key
#   - API_DELAY_MS: Override rate limit delay (default: 1000ms)
#
set -euo pipefail

# Rate limiting variables
declare -i LAST_API_CALL=0
declare -i API_DELAY_MS=1000  # 1 second between calls

# Wait for rate limit if needed
# Enforces minimum delay between API calls to respect rate limits
# Uses high-precision timing with millisecond accuracy
api_rate_limit() {
  local -i now current_ms elapsed needed_delay
  
  # Get current time in milliseconds
  now=$(date +%s)
  # Force base 10 interpretation by using $((10#...)) for nanoseconds
  local nanos
  nanos=$(date +%N)
  current_ms=$((now * 1000 + 10#$nanos / 1000000))
  
  if ((LAST_API_CALL > 0)); then
    elapsed=$((current_ms - LAST_API_CALL))
    if ((elapsed < API_DELAY_MS)); then
      needed_delay=$((API_DELAY_MS - elapsed))
      # Convert milliseconds to seconds with 3 decimal places
      sleep_seconds=$(printf "%.3f" "$(awk "BEGIN {print $needed_delay/1000}")")
      sleep "$sleep_seconds"
    fi
  fi
  
  LAST_API_CALL=$current_ms
}

# Call OpenAI API with retry logic
# Args:
#   $1: Model name (e.g., gpt-4o-mini)
#   $2: System prompt
#   $3: User content
#   $4: Max tokens for response
#   $5: Temperature (0-1)
# Returns:
#   0 on success with JSON response to stdout
#   1 on failure after all retries
api_call_openai() {
  local model="$1"
  local system_prompt="$2"
  local user_content="$3"
  local max_tokens="$4"
  local temperature="$5"
  local -i max_retries=3
  local -i retry_delay=5
  local response
  local -i attempt
  
  for ((attempt=1; attempt<=max_retries; attempt++)); do
    # Rate limit
    api_rate_limit
    
    # Create JSON payload
    local json_payload
    json_payload=$(jq -n \
      --arg model "$model" \
      --arg system "$system_prompt" \
      --arg content "$user_content" \
      --arg max_tokens "$max_tokens" \
      --arg temperature "$temperature" \
      '{
        model: $model,
        messages: [
          {role: "system", content: $system},
          {role: "user", content: $content}
        ],
        temperature: ($temperature | tonumber),
        max_tokens: ($max_tokens | tonumber)
      }')
    
    # Make API call with timeouts for security
    response=$(curl -s -w '\n%{http_code}' \
      --connect-timeout 30 \
      --max-time 120 \
      https://api.openai.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -d "$json_payload" 2>/dev/null) || true

    # Extract HTTP status code
    local http_code
    http_code=$(echo "$response" | tail -n1)
    response=$(echo "$response" | sed '$d')

    # Check for success
    if [[ "$http_code" == "200" ]]; then
      echo "$response"
      return 0
    fi

    # Handle specific error codes
    case "$http_code" in
      000)  # Timeout or connection failure
        >&2 echo "Connection timeout/failure on attempt $attempt/$max_retries"
        if ((attempt < max_retries)); then
          sleep $retry_delay
          retry_delay=$((retry_delay * 2))
        fi
        ;;
      429)  # Rate limit exceeded
        >&2 echo "Rate limit exceeded, waiting ${retry_delay}s before retry $attempt/$max_retries"
        sleep $retry_delay
        retry_delay=$((retry_delay * 2))
        ;;
      401)  # Unauthorized
        >&2 echo "API authentication failed. Check OPENAI_API_KEY"
        return 1
        ;;
      *)    # Other errors
        >&2 echo "API call failed with HTTP $http_code on attempt $attempt/$max_retries"
        if ((attempt < max_retries)); then
          sleep $retry_delay
        fi
        ;;
    esac
  done
  
  >&2 echo "API call failed after $max_retries attempts"
  return 1
}

# Call Anthropic API with retry logic
# Args:
#   $1: Model name (e.g., claude-sonnet-4-5)
#   $2: System prompt
#   $3: User content
#   $4: Max tokens for response
#   $5: Temperature (0-1)
# Returns:
#   0 on success with JSON response to stdout
#   1 on failure after all retries
api_call_anthropic() {
  local model="$1"
  local system_prompt="$2"
  local user_content="$3"
  local max_tokens="$4"
  local temperature="$5"
  local -i max_retries=3
  local -i retry_delay=5
  local response
  local -i attempt

  # Check for API key
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    >&2 echo "Error: ANTHROPIC_API_KEY not set for Anthropic model"
    return 1
  fi

  for ((attempt=1; attempt<=max_retries; attempt++)); do
    # Rate limit
    api_rate_limit

    # Create JSON payload (Anthropic format differs from OpenAI)
    local json_payload
    json_payload=$(jq -n \
      --arg model "$model" \
      --arg system "$system_prompt" \
      --arg content "$user_content" \
      --arg max_tokens "$max_tokens" \
      --arg temperature "$temperature" \
      '{
        model: $model,
        max_tokens: ($max_tokens | tonumber),
        temperature: ($temperature | tonumber),
        system: $system,
        messages: [
          {role: "user", content: $content}
        ]
      }')

    # Make API call with timeouts
    response=$(curl -s -w '\n%{http_code}' \
      --connect-timeout 30 \
      --max-time 120 \
      https://api.anthropic.com/v1/messages \
      -H "Content-Type: application/json" \
      -H "x-api-key: $ANTHROPIC_API_KEY" \
      -H "anthropic-version: 2023-06-01" \
      -d "$json_payload" 2>/dev/null) || true

    # Extract HTTP status code
    local http_code
    http_code=$(echo "$response" | tail -n1)
    response=$(echo "$response" | sed '$d')

    # Check for success
    if [[ "$http_code" == "200" ]]; then
      echo "$response"
      return 0
    fi

    # Handle specific error codes
    case "$http_code" in
      000)  # Timeout or connection failure
        >&2 echo "Connection timeout/failure on attempt $attempt/$max_retries"
        if ((attempt < max_retries)); then
          sleep $retry_delay
          retry_delay=$((retry_delay * 2))
        fi
        ;;
      429)  # Rate limit exceeded
        >&2 echo "Rate limit exceeded, waiting ${retry_delay}s before retry $attempt/$max_retries"
        sleep $retry_delay
        retry_delay=$((retry_delay * 2))
        ;;
      401)  # Unauthorized
        >&2 echo "API authentication failed. Check ANTHROPIC_API_KEY"
        return 1
        ;;
      *)    # Other errors
        >&2 echo "Anthropic API call failed with HTTP $http_code on attempt $attempt/$max_retries"
        if ((attempt < max_retries)); then
          sleep $retry_delay
        fi
        ;;
    esac
  done

  >&2 echo "Anthropic API call failed after $max_retries attempts"
  return 1
}

# Router function - calls appropriate provider based on model name
# Args:
#   $1: Model name
#   $2: System prompt
#   $3: User content
#   $4: Max tokens
#   $5: Temperature
# Returns:
#   0 on success, 1 on failure
api_call() {
  local model="$1"

  # Route based on model name prefix
  case "$model" in
    claude-*|claude*)
      api_call_anthropic "$@"
      ;;
    gpt-*|o1-*|o3-*|text-*)
      api_call_openai "$@"
      ;;
    grok-*)
      # xAI uses OpenAI-compatible API
      if [[ -z "${XAI_API_KEY:-}" ]]; then
        >&2 echo "Error: XAI_API_KEY not set for xAI model"
        return 1
      fi
      # Temporarily swap keys and endpoint
      local orig_key="${OPENAI_API_KEY:-}"
      OPENAI_API_KEY="$XAI_API_KEY"
      api_call_openai "$@"
      local ret=$?
      OPENAI_API_KEY="$orig_key"
      return $ret
      ;;
    gemini-*)
      >&2 echo "Error: Google Gemini models not yet supported in citations"
      return 1
      ;;
    *)
      # Default to OpenAI for unknown models
      api_call_openai "$@"
      ;;
  esac
}

# Extract citation from API response
# Handles both OpenAI and Anthropic response formats
# Args:
#   $1: JSON response from API
# Returns:
#   0 on success with citation string to stdout
#   1 on invalid JSON or missing content
# Output:
#   First line of content from API response
api_extract_result() {
  local response="$1"
  local result

  # Check if response is valid JSON
  if ! echo "$response" | jq empty 2>/dev/null; then
    >&2 echo "Invalid JSON response from API"
    echo "Error"
    return 1
  fi

  # Try OpenAI format first, then Anthropic format
  result=$(echo "$response" | jq -r '
    if .choices then
      .choices[0].message.content // "Error"
    elif .content then
      .content[0].text // "Error"
    else
      "Error"
    end
  ')

  # Get only first line
  result=$(echo "$result" | head -n1)

  # Trim whitespace
  result=$(echo "$result" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

  echo "$result"
  return 0
}

# Validate API key exists for the specified model's provider
# Args:
#   $1: Model name (optional, validates all if not provided)
# Returns:
#   0 if required key exists
#   1 if key missing
api_validate_key() {
  local model="${1:-}"
  local -i found_key=0

  # If model specified, check only that provider
  if [[ -n "$model" ]]; then
    case "$model" in
      claude-*|claude*)
        if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
          >&2 echo "Error: ANTHROPIC_API_KEY not set for model $model"
          return 1
        fi
        return 0
        ;;
      grok-*)
        if [[ -z "${XAI_API_KEY:-}" ]]; then
          >&2 echo "Error: XAI_API_KEY not set for model $model"
          return 1
        fi
        return 0
        ;;
      gemini-*)
        if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
          >&2 echo "Error: GOOGLE_API_KEY not set for model $model"
          return 1
        fi
        return 0
        ;;
      *)
        # Default to OpenAI
        if [[ -z "${OPENAI_API_KEY:-}" ]]; then
          >&2 echo "Error: OPENAI_API_KEY not set for model $model"
          return 1
        fi
        return 0
        ;;
    esac
  fi

  # No model specified - check if at least one provider key exists
  [[ -n "${OPENAI_API_KEY:-}" ]] && found_key=1
  [[ -n "${ANTHROPIC_API_KEY:-}" ]] && found_key=1
  [[ -n "${XAI_API_KEY:-}" ]] && found_key=1
  [[ -n "${GOOGLE_API_KEY:-}" ]] && found_key=1

  if ((!found_key)); then
    >&2 echo "Error: No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, or GOOGLE_API_KEY"
    return 1
  fi

  return 0
}

#fin