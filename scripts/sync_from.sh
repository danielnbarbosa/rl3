#!/bin/bash
# sync files from Cloud GPU instance to local
# works on lambda and paperspace

# example:
# ../scripts/sync_from.sh ubuntu@104.171.202.193

USER_IP=$1
REMOTE_DIR=$(cat current_training_run)
ENV=$(basename "$(pwd)")
LOCAL_DIR="/Users/daniel/src/rl3/$ENV/training_runs/"
#LOCAL_DIR="/Users/daniel/src/github/danielnbarbosa/rl3/$ENV/training_runs"

echo "Syncing $ENV $REMOTE_DIR"
rsync -av --exclude train.log --exclude train_steps_*.pth --exclude *.pkl ${USER_IP}:${REMOTE_DIR} $LOCAL_DIR
#rsync -av ${USER_IP}:${REMOTE_DIR} $LOCAL_DIR
