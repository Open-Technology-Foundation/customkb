#!/bin/bash
# Mock OpenAI API server for testing

set -euo pipefail

# Default configuration
PORT="${1:-8888}"
RESPONSES_DIR="${2:-$(dirname "$0")/../fixtures/api_responses}"
LOG_FILE="/tmp/mock_openai_api.log"

# Colors
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
NOCOLOR=$'\033[0m'

# Help message
show_help() {
  cat << EOF
Usage: $(basename "$0") [PORT] [RESPONSES_DIR]

Start a mock OpenAI API server for testing.

Arguments:
  PORT           Port to listen on (default: 8888)
  RESPONSES_DIR  Directory containing response JSON files (default: ../fixtures/api_responses)

Options:
  -h, --help     Show this help message
  --daemon       Run in background

The server responds to:
  GET  /health                    - Health check
  POST /v1/chat/completions       - Mock chat completions endpoint

Response behavior:
  - Looks for specific patterns in request to determine response
  - Returns appropriate test responses or errors
  - Logs all requests to $LOG_FILE

EOF
}

# Parse arguments
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "${RED}Error: Port $PORT is already in use${NOCOLOR}"
  exit 1
fi

# Simple HTTP server using netcat and bash
start_server() {
  echo "${GREEN}Starting mock OpenAI API server on port $PORT${NOCOLOR}"
  echo "Responses directory: $RESPONSES_DIR"
  echo "Log file: $LOG_FILE"
  echo "${YELLOW}Press Ctrl+C to stop${NOCOLOR}"
  echo
  
  # Create log file
  : > "$LOG_FILE"
  
  while true; do
    # Use netcat to listen for connections
    {
      # Read request
      local request=""
      local content_length=0
      local body=""
      
      while IFS= read -r line; do
        line="${line%%$'\r'}"
        
        if [[ -z "$line" ]]; then
          # Empty line marks end of headers
          break
        fi
        
        request+="$line"$'\n'
        
        # Extract content length
        if [[ "$line" =~ ^[Cc]ontent-[Ll]ength:\ *([0-9]+) ]]; then
          content_length="${BASH_REMATCH[1]}"
        fi
      done
      
      # Read body if present
      if ((content_length > 0)); then
        body=$(head -c "$content_length")
      fi
      
      # Log request
      {
        echo "=== Request at $(date) ==="
        echo "$request"
        echo "Body: $body"
        echo
      } >> "$LOG_FILE"
      
      # Determine response based on request
      local response_file=""
      local status="200 OK"
      local response_body=""
      
      if [[ "$request" =~ GET\ /health ]]; then
        # Health check
        response_body='{"status": "ok", "server": "mock-openai-api"}'
        
      elif [[ "$request" =~ POST\ /v1/chat/completions ]]; then
        # Chat completions endpoint
        
        # Determine response based on body content
        if [[ "$body" =~ rate_limit_test ]]; then
          status="429 Too Many Requests"
          response_file="$RESPONSES_DIR/rate_limit_error.json"
        elif [[ "$body" =~ auth_test ]]; then
          status="401 Unauthorized"
          response_file="$RESPONSES_DIR/auth_error.json"
        elif [[ "$body" =~ error_test ]]; then
          status="500 Internal Server Error"
          response_file="$RESPONSES_DIR/server_error.json"
        elif [[ "$body" =~ not_found_test ]]; then
          response_file="$RESPONSES_DIR/not_found_response.json"
        else
          # Default response
          response_file="$RESPONSES_DIR/standard_response.json"
        fi
        
        # Read response file
        if [[ -f "$response_file" ]]; then
          response_body=$(cat "$response_file")
        else
          response_body='{"error": {"message": "Mock response not found", "type": "mock_error"}}'
          status="500 Internal Server Error"
        fi
        
      else:
        # Unknown endpoint
        status="404 Not Found"
        response_body='{"error": {"message": "Not found", "type": "not_found"}}'
      fi
      
      # Send response
      local content_length=${#response_body}
      
      echo -ne "HTTP/1.1 $status\r\n"
      echo -ne "Content-Type: application/json\r\n"
      echo -ne "Content-Length: $content_length\r\n"
      echo -ne "Connection: close\r\n"
      echo -ne "\r\n"
      echo -n "$response_body"
      
      # Log response
      {
        echo "=== Response ==="
        echo "Status: $status"
        echo "Body: $response_body"
        echo "===="
        echo
      } >> "$LOG_FILE"
      
    } | nc -l -p "$PORT" -q 1
    
    # Small delay to prevent CPU spinning
    sleep 0.1
  done
}

# Trap to clean up on exit
cleanup() {
  echo
  echo "${YELLOW}Shutting down mock API server${NOCOLOR}"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Start server
start_server

#fin