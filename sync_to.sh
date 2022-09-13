#!/bin/bash
# sync files from local to cloud GPU instance
# works on lambda and paperspace

# example
# ../sync_to.sh ubuntu@104.171.203.82

USER_IP=$1
DIR="/Users/daniel/src/rl3/"
ENV="breakout"

scp $DIR/train.sh $DIR/install_deps.sh $DIR/$ENV/dqn.py ${USER_IP}:

# on cloud instance:
# ./install_deps.sh
# ./train.sh &
# now safe to disconnect SSH connection
# monitor: `tail -f train-summary.log`
# stop training: `pgrep -f train`