#!/bin/bash
# install dependencies on Lambda GPU cloud instances
# works on lambda and paperspace

# example:
# ./install_deps.sh

sudo apt-get update
sudo apt-get install -y python3-dev cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
pip install --upgrade pip
pip install torchinfo opencv-python vizdoom==1.1.11

# add pub keys
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG/4SlsKtQM3Vr0NplhST4P+JxbVKaPMGWHXzFStSfmE' >> ~/.ssh/authorized_keys
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDWxEkVuF2UU9xWRcjne53AzpdgwB+YlpcKzcAo5R6s7/xI2a5QyGhmMynGzM+82Ye8Vv6mBbzi3tVceAR6xDcYMOyFd529P545nrNbej8r/nHCCxV4z/KeYuM4WSghLBAkc2rJl9GQOUPSBQZokeHLTo0DfsGWFwJa3vf3xLswUYNLQBnDhmhUr6eE/3XU+5QJ0k5U30e5KPZcSAA6vd5q8EXUEUfkh0evMzMEVWWDPJSvf6FUxzy1OdxsF92sxUEvAMioyogn4z+ew/+4xpQk4feWkV052f8N+S6FX/4oiwIBAnWK96gHh1okGBhEo+7G9WHzzwWrHc63fJkAR4Xe5x4x7CAkOId2uuK4iVf8zjlTyhDNZ6b1V/Z4NOG/bRKmQyZ+LNXDkNJL9sLoQ2YbuKjMdo8H3zec5sFTeHpUqwluYSYmlZ1GFKsbZRawhBfiy3Tp5aAzjQ/uUFvf6mtgcOIv/FDEt/Jp3kvpjNmeDjqXxYFJv17tjMPxLHXUrDs=' >> ~/.ssh/authorized_keys