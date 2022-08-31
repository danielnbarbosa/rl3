#!/bin/bash
# sync files from local to cloud GPU instance
# works on lambda and paperspace

USER_IP=$1
DIR="/Users/daniel/src/rl_class/"
ENV="super-mario-bros"

scp $DIR/train.sh $DIR/install_deps.sh $DIR/rl2/$ENV/dqn.py $DIR/rl2/$ENV/dqn_wrappers.py ${USER_IP}:

# on cloud instance:
# run install_deps.sh
# run train.sh &
# now safe to disconnect SSH connection
# monitor with tail -f train-summary.log