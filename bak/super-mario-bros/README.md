## Train
Trained on Lambda Cloud A6000.
Environment is 'SuperMarioBrosRandomStages-v0', stages=['1-1', '1-2', '1-3', '1-4']
This randomizes all 4 levels from world 1.

```
Ep: 59000	AvgR: 842.94	BestAvgR: 904.76	Eps: 0.1	RepBuf: 1000000	Steps:207	TotSteps: 7119548	RunTime: 2s (0/2/1)	TotRunTime: 83539s	Steps/s: 85
```

Total steps: 7_119_548
Total time: 83_539 seconds (23 hours)

To see tensorboard output:  `tensorboard --logdir=saved_training_runs/cuda-2022-09-01-00:49:22.635376/runs`

To restart training from latest saved model: `python -u dqn.py -m train -f saved_training_runs/cuda-2022-09-01-00:49:22.635376/models/latest.pth`


## Eval
For demo run `eval_best.sh`
Mario makes some progress but often gets stuck.  Rarely finishes a level.
Average Episode Reward across 10 episodes: 1151.0


## Play
To play as human, run `play.sh`


## Ideas
Decrease hyper parameters, like replay memory size and target sync.
Focus on a single level.
Try more complex action space (especially for stage 1-2).
Do reward hacking.
Train longer.