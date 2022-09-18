#!/bin/bash
# install dependencies on Lambda GPU cloud instances
# works on lambda and paperspace

sudo apt-get update
sudo apt-get install -y swig python3-dev
pip install --upgrade pip
pip install gym==0.25.2 gym[box2d] gym[accept-rom-license] gym[atari] gym_super_mario_bros torchinfo opencv-python
# add pub key
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG/4SlsKtQM3Vr0NplhST4P+JxbVKaPMGWHXzFStSfmE' >> ~/.ssh/authorized_keys