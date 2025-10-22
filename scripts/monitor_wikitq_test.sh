#!/bin/bash
# Monitor WikiTQ AILS test progress

LOG_FILE="logs/wikitq_ails_10_fixed.log"

echo "=== WikiTQ AILS Test Progress Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

while true; do
    clear
    echo "=== WikiTQ AILS Test Progress ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""

    # Check if process is still running
    if ps -p 1103179 > /dev/null 2>&1; then
        echo "Status: ✓ RUNNING (PID: 1103179)"
    else
        echo "Status: ✓ COMPLETED or ✗ TERMINATED"
    fi
    echo ""

    # Show latest results
    echo "=== Latest Results ==="
    tail -15 "$LOG_FILE" | grep -E "Results:|Execution Success|Answer Correct|Avg Iterations|TEST|Configuration|IMPROVEMENTS"
    echo ""

    # Show summary if completed
    if grep -q "SUMMARY COMPARISON" "$LOG_FILE"; then
        echo "=== FINAL SUMMARY ==="
        grep -A 10 "SUMMARY COMPARISON" "$LOG_FILE" | tail -7
    fi

    sleep 10
done
