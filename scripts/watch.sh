#!/bin/bash
# watch agent play using latest model

# example:
# ../scripts/watch.sh

DIR=$(cat current_training_run)

echo "Using latest model in $DIR"
python dqn.py -m eval -f $DIR/models/latest.pth -r human
