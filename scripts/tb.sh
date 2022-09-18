#!/bin/bash
# start tensorboard on current training run

# example:
# ../scripts/tb.sh

DIR=$(cat current_training_run)

echo "Loading logs for $DIR"
tensorboard --logdir $DIR/runs