This is a project recreating the paper "Dynamic Quantization Network for Efficient Neural Network Inference" (DQNet) on CIFAR10, and attempting to improve upon it, specifically exploring ideas to improve latency. 

I have tried to make my code and experiments as easy to reproduce as possible, but please understand that as experimentation was the goal, there is not a clear end to end which will run everything ive done. Rather, you can recreate the baselines, and experiments from each notebook individually.

Also, since the code was run on colab, the code used to make the final results may have been slightly modified within colab. To the best of my knowledge, the code is accurate to the final used, but there may be minor discrepancies.

You can run any of the experiments or baseline or DQbase notebooksto train the models and save the checkpoints. 

DQbase is the Dynamic Quantization ResNet 20 Baseline., baseline is just Resnet 20, experiments 1,2,3 are as detailed within the report.

I have used G4 gpus on google colab to run everything. Latency measurements were done on the same, recommend to follow if you wish to reproduce. 

Please note that training takes a while

Please note in several places, bitFlops are presented as a ratio with fp32max. Hence the numbers may not match those reported within the paper. I converted them to later better match the original DQNet paper for comparisons sake, but kept the code as is as its easier to compaer models while experimenting.