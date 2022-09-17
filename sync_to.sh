#!/bin/bash
# sync files from local to cloud GPU instance
# works on lambda and paperspace

# example
# ../sync_to.sh ubuntu@104.171.203.82

USER_IP=$1
DIR="/Users/daniel/src/rl3/"
ENV=$(basename "$(pwd)")

echo "Syncing $ENV"
scp $DIR/*.sh $DIR/$ENV/*.py ${USER_IP}:

# on cloud instance:
# ./install_deps.sh
# ./train.sh &
# now safe to disconnect SSH connection
# ./watch.sh  # to monitor logs