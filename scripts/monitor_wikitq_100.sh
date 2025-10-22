#!/bin/bash
# Automated progress checker for WikiTQ 100-sample test

LOG_FILE="logs/wikitq_100_ails.log"
PID=1147818

echo "=== WikiTQ 100-Sample Test Monitor ==="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo ""

# Check every 5 minutes until completion
while true; do
    CURRENT_TIME=$(date '+%H:%M:%S')

    # Check if process is still running
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "[$CURRENT_TIME] ✓ Process completed or terminated"
        break
    fi

    # Extract progress information
    BASELINE_PROG=$(grep -oP 'Baseline:\s+\K\d+%' "$LOG_FILE" | tail -1)
    AILS_PROG=$(grep -oP 'AILS Zero-Shot:\s+\K\d+%' "$LOG_FILE" | tail -1)

    echo "[$CURRENT_TIME] Progress:"
    if [ -n "$BASELINE_PROG" ]; then
        echo "  Baseline: $BASELINE_PROG"
    fi
    if [ -n "$AILS_PROG" ]; then
        echo "  AILS Zero-Shot: $AILS_PROG"
    fi

    # Check if completed
    if grep -q "FINAL COMPARISON" "$LOG_FILE"; then
        echo "[$CURRENT_TIME] ✓ Test COMPLETED!"
        echo ""
        echo "=== FINAL RESULTS ==="
        grep -A 10 "FINAL COMPARISON" "$LOG_FILE"
        break
    fi

    # Wait 5 minutes before next check
    sleep 300
done

echo ""
echo "Monitor ended: $(date '+%Y-%m-%d %H:%M:%S')"
