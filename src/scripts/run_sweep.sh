#!/bin/bash

NUM_SEEDS=15
NUM_RNGSEEDS=15   # how many independent experiments to launch

for ((i=1; i<=NUM_RNGSEEDS; i++))
do
    RNGSEED=$(shuf -i 1-1000000 -n 1)
    python src/scripts/run_experiment1.py \
        --numSeeds ${NUM_SEEDS} \
        --rngSeed ${RNGSEED} \
        --hiddenSize 64 \
        --flushEvery 5 &
done

wait
echo "All experiments completed."
# End of file src/scripts/run_sweep.sh

