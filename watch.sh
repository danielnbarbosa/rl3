#!/bin/bash
# watch agent play using latest model

python dqn.py -m eval -f $(cat current_training_run)/models/latest.pth -r human
