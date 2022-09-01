#!/bin/bash
# install dependencies on Lambda GPU cloud instances
# works on lambda and paperspace

sudo apt-get update
sudo apt-get install -y swig python3-dev
pip install --upgrade pip
pip install gym gym[box2d] gym[accept-rom-license] gym[atari] gym_super_mario_bros torchinfo opencv-python