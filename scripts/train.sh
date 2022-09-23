#!/bin/bash
# helpful for training in the background on cloud GPU instances
# works on lambda and paperspace

# example:
# ./train.sh &

# lambda
if [ $(whoami) == "ubuntu" ]
    then PYTHON="python"
# paperspace
elif [ $(whoami) == "paperspace" ]
    then PYTHON="python3"
fi
date > train-summary.log
stdbuf -oL $PYTHON -u dqn.py -m train | tee -a train-summary.log