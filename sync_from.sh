#!/bin/bash
# sync files from Cloud GPU instance to local
# works on lambda and paperspace

USER_IP=$1
REMOTE_DIR=$2
ENV="atari"
LOCAL_DIR="/Users/daniel/src/rl3/$ENV/training_runs/"

rsync -av --exclude train.log --exclude episode_*.pth  ${USER_IP}:${REMOTE_DIR} $LOCAL_DIR
