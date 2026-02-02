#!/bin/bash
set -euo pipefail

# Monitor GPU usage during CustomKB query

declare config="${1:-seculardharma}"
declare query="${2:-What is dharma?}"
if [[ -n "${1:-}" ]]; then
  shift
fi
if [[ -n "${2:-}" ]]; then
  shift
fi


echo "GPU Monitoring for CustomKB Query"
echo "================================="
echo "Query: $query"
echo "Config: $config"
echo ""

# Start GPU monitoring in background
nvidia-smi dmon -s u -d 1 > /tmp/gpu_usage.log &
declare -r monitor_pid=$!

# Run the query
echo "Running query..."
time customkb query "$config" "$query" "$@" > /tmp/query_output.txt 2>&1

# Stop monitoring
sleep 2
kill $monitor_pid 2>/dev/null || true

# Analyze GPU usage
echo ""
echo "GPU Usage Statistics:"
echo "--------------------"

# Skip header and calculate stats
awk 'NR>2 && $2 ~ /^[0-9]+$/ {
    util += $2
    mem += $3
    count++
} 
END {
    if (count > 0) {
        printf "Average GPU Utilization: %.1f%%\n", util/count
        printf "Average Memory Usage: %.1f%%\n", mem/count
        printf "Peak values from monitoring\n"
    }
}' /tmp/gpu_usage.log

# Show peak usage
echo ""
echo "Peak GPU Utilization: $(awk 'NR>2 && $2 ~ /^[0-9]+$/ {if($2>max)max=$2} END{print max}' /tmp/gpu_usage.log)%"
echo "Peak Memory Usage: $(awk 'NR>2 && $3 ~ /^[0-9]+$/ {if($3>max)max=$3} END{print max}' /tmp/gpu_usage.log)%"

# Extract timing info
echo ""
echo "Query Performance:"
echo "-----------------"
grep -E "(Elapsed|BM25 results|Reranking)" /tmp/query_output.txt | grep -v "bash" | tail -10

# Cleanup
rm -f /tmp/gpu_usage.log /tmp/query_output.txt

#fin