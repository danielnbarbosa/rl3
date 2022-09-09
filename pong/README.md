## Train
Trained on Lambda Cloud A6000.

```
Ep: 4000	AvgR: 16.42	BestAvgR: 16.43	Eps: 0.1	RepBuf: 1000000	Steps:1913	TotSteps: 8025830	RunTime: 9s (2/2/5)	TotRunTime: 37545s	Steps/s: 211
```


Total steps: 8_025_830
Total time: 37_545 seconds (10 hours)

To see tensorboard output:  `tensorboard --logdir=saved_training_runs/cuda-2022-09-02-01:00:20.826436/runs`

## Eval
For demo run `eval_best.sh`
Agent is near perfect.
Average Episode Reward across 10 episodes: 20.5


## Play
To play as human, run `python play.py`
