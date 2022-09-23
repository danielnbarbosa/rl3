#!/bin/bash
# install dependencies on Lambda GPU cloud instances
# works on lambda and paperspace

# example:
# ./install_deps.sh

sudo apt-get update
sudo apt-get install -y python3-dev cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
pip install --upgrade pip
pip install torchinfo opencv-python vizdoom==1.1.11

# add pub key
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG/4SlsKtQM3Vr0NplhST4P+JxbVKaPMGWHXzFStSfmE' >> ~/.ssh/authorized_keys
