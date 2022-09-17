#!/bin/bash
# helpful for training in the background on cloud GPU instances
# works on lambda and paperspace

# stop prior run
pkill -f train

# lambda
if [ $(whoami) == "ubuntu" ]
    then stdbuf -oL python -u dqn.py -m train | tee train-summary.log
# paperspace
elif [ $(whoami) == "paperspace" ]
    then stdbuf -oL python3 -u dqn.py -m train | tee train-summary.log
fi
