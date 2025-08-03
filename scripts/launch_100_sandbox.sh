#!/bin/bash

# Stage 3 100-Iteration Sandbox Launcher
# Based on the user's playbook requirements

set -e

# Configuration
RUN_ID="sandbox_100_$(date +%Y%m%d_%H%M)"
ITERS=100
LAMBDA=0.4
BATCH_SIZE=10  # Run in batches to avoid overwhelming the system

echo "🎯 Stage 3 100-Iteration Sandbox"
echo "=================================="
echo "Run ID: $RUN_ID"
echo "Iterations: $ITERS"
echo "Lambda DiffDock: $LAMBDA"
echo "Batch size: $BATCH_SIZE"
echo "Start time: $(date)"
echo ""

# Function to trigger a batch of iterations
trigger_batch() {
    local start_iter=$1
    local end_iter=$2
    local batch_num=$3
    
    echo "🚀 Launching batch $batch_num (iterations $start_iter-$end_iter)"
    
    for i in $(seq $start_iter $end_iter); do
        run_id="${RUN_ID}_iter_${i:03d}"
        echo "  Triggering iteration $i/100: $run_id"
        
        docker compose exec -T airflow-worker \
            airflow dags trigger dit_uq_stage3 \
            --run-id "$run_id" \
            --conf "{\"iters\":1,\"lambda_diffdock\":$LAMBDA}"
        
        # Small delay between triggers
        sleep 1
    done
    
    echo "✅ Batch $batch_num launched"
}

# Function to monitor progress
monitor_progress() {
    echo ""
    echo "📊 Monitoring Progress"
    echo "====================="
    
    while true; do
        # Count successful runs
        success_count=$(docker compose exec -T airflow-worker \
            airflow dags list-runs --dag-id dit_uq_stage3 2>/dev/null | \
            grep "$RUN_ID" | grep "success" | wc -l)
        
        # Count total runs
        total_count=$(docker compose exec -T airflow-worker \
            airflow dags list-runs --dag-id dit_uq_stage3 2>/dev/null | \
            grep "$RUN_ID" | wc -l)
        
        # Count running runs
        running_count=$(docker compose exec -T airflow-worker \
            airflow dags list-runs --dag-id dit_uq_stage3 2>/dev/null | \
            grep "$RUN_ID" | grep "running" | wc -l)
        
        echo "$(date '+%H:%M:%S') - Progress: $success_count/$ITERS completed, $running_count running"
        
        if [ $success_count -eq $ITERS ]; then
            echo "🎉 All iterations completed!"
            break
        fi
        
        sleep 30
    done
}

# Main execution
echo "Starting 100-iteration sandbox..."

# Launch iterations in batches
for batch in $(seq 1 $((ITERS/BATCH_SIZE))); do
    start_iter=$(((batch-1)*BATCH_SIZE + 1))
    end_iter=$((batch*BATCH_SIZE))
    
    if [ $end_iter -gt $ITERS ]; then
        end_iter=$ITERS
    fi
    
    trigger_batch $start_iter $end_iter $batch
    
    # Wait a bit between batches
    if [ $batch -lt $((ITERS/BATCH_SIZE)) ]; then
        echo "⏳ Waiting 10 seconds before next batch..."
        sleep 10
    fi
done

echo ""
echo "✅ All iterations launched!"
echo "📊 Starting progress monitoring..."

# Monitor progress in background
monitor_progress &

# Wait for completion
wait

echo ""
echo "🎉 100-Iteration Sandbox Complete!"
echo "=================================="
echo "Run ID: $RUN_ID"
echo "End time: $(date)"
echo ""
echo "📁 Results available in data/stage3_results.parquet"
echo "📊 Check Airflow UI for detailed task logs"
echo ""
echo "Next steps:"
echo "1. Analyze results with physics-aware metrics"
echo "2. Generate Pareto plots and hypervolume analysis"
echo "3. Prepare for λ-sweep ablation study" 