#!/bin/bash
# start tensorboard on current training run

LOG_DIR=$(cat current_training_run)

tensorboard --logdir $LOG_DIR/runs