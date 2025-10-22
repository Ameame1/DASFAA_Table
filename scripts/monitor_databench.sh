#!/bin/bash
# Monitor DataBench 100-sample evaluation progress

LOG_FILE="logs/databench_100_eval.log"

echo "=========================================="
echo "DataBench 100-Sample Evaluation Monitor"
echo "=========================================="
echo ""

# Check if process is running
if pgrep -f "evaluate_databench.py --num_samples 100" > /dev/null; then
    echo "✓ Evaluation is running"
    PID=$(pgrep -f "evaluate_databench.py --num_samples 100")
    echo "  PID: $PID"
else
    echo "✗ Evaluation not running"
    exit 1
fi

echo ""
echo "Latest progress:"
echo "----------------------------------------"

# Show last 30 lines
tail -30 "$LOG_FILE"

echo ""
echo "----------------------------------------"
echo "Statistics so far:"

# Count completed samples
COMPLETED=$(grep -c "Sample [0-9]*/100" "$LOG_FILE" 2>/dev/null || echo "0")
echo "  Samples completed: $COMPLETED/100"

# Count correct answers
CORRECT=$(grep -c "Correctness: ✓ Correct" "$LOG_FILE" 2>/dev/null || echo "0")
echo "  Correct answers: $CORRECT"

# Count execution success
SUCCESS=$(grep -c "Execution: ✓ Success" "$LOG_FILE" 2>/dev/null || echo "0")
echo "  Execution success: $SUCCESS"

if [ $COMPLETED -gt 0 ]; then
    ACCURACY=$((CORRECT * 100 / COMPLETED))
    EXEC_RATE=$((SUCCESS * 100 / COMPLETED))
    echo "  Current accuracy: $ACCURACY%"
    echo "  Current exec rate: $EXEC_RATE%"
fi

echo ""
echo "To monitor continuously: tail -f $LOG_FILE"
echo "To stop: kill $PID"
