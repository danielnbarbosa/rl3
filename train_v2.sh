#!/bin/bash
# helpful for training in the background on cloud GPU instances
# works on lambda and paperspace

# lambda
if [ $(whoami) == "ubuntu" ]
    then stdbuf -oL python -u dqn_v2.py -m train | tee train-summary.log
# paperspace
elif [ $(whoami) == "paperspace" ]
    then stdbuf -oL python3 -u dqn_v2.py -m train | tee train-summary.log
fi
