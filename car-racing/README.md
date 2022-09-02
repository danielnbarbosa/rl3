## Train
Trained on M1 for a while then stopped and restarted training due to memory leak on M1.

```
Ep: 3212	AvgR: 701.96	BestAvgR: 772.1	    Eps: 0.1	RepBuf: 100000	Steps:250	TotSteps: 793982	RunTime: 11s (0/5/6)	TotRunTime: 34797s	Steps/s: 22
Ep: 3142	AvgR: 741.5	    BestAvgR: 781.53	Eps: 0.1	RepBuf: 100000	Steps:250	TotSteps: 780278	RunTime: 11s (1/5/6)	TotRunTime: 44425s	Steps/s: 22
```

Total steps: 1_574_260
Total time: 79_222 seconds (22 hours)

To see tensorboard output:  `tensorboard --logdir=saved_training_runs/cpu-2022-09-01-06:47:36.812964/runs`


## Eval
For demo run `eval_best.sh`
Car does very well.
Average Episode Reward across 10 episodes: 847.3863189352514


## Play
To play as human, run `play.sh`