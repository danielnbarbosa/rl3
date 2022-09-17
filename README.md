My third iteration of RL code.  This time focusing on learning to play games from pixels.

## train on cloud instance:
# from local machine in env specific dir (e.g. atari)
../sync_to.sh <USER@IP>  # to upload training script
ssh <USER@IP>
# on cloud instance
./install_deps.sh
./train.sh &
# now safe to disconnect SSH connection
./watch.sh  # to monitor logs
# back on local machine
./set.sh <training_run_path>
../sync_from.sh <USER@IP>  # to download models and metrics
../tb.sh  # to start tensorboard
# to stop training:
pkill -f train



## train
to train from scratch: `python dqn.py -m train`
to train from a saved model (loads corresponding replay buffer as well): `python dqn.py -m train -f <PATH TO .PTH MODEL FILE>`

# eval
to evaluate a trained model: `python dqn.py -m eval -f <PATH TO .PTH MODEL FILE>`
to evaluate a trained model and render to the screen: `python dqn.py -m eval -f <PATH TO .PTH MODEL FILE> -r human`
to evaluate a trained model and render to mp4 video file: `python dqn.py -m eval -f <PATH TO .PTH MODEL FILE> -r video`


## evolution
DQN code evolved in the following order: car-racing -> super-mario-bros -> pong -> breakout