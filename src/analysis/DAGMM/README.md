Recommended Setting (Ok, I only tested in the following setting):
1. tensorflow 1.x (GPU version is encouraged for fast speed)
2. python 3.x, including numpy

Command: python runDagmm.py --dataset <dataset_name>
where <dataset_name> could be kddcup, thyroid, arrhythmia, or kddcuprev. They are the four datasets used in the experiment of our paper.

Example: python runDagmm.py --dataset kddcup

After executing this command, the following are going to happen.
(1) The target dataset will be randomly partitioned into training and testing set (50/50);
(2) DAGMM will be trained based on training data;
(3) Accuracy will be evaluated and printed out against testing data;
(4) The above 3 steps will be repeated multiple times (hard coded as 20, but you can change), and the average accuracy will be reported finally.
