# Performance-Estimation-Problems-for-distributed-optimization

This repository contains the codes used in my thesis:
    
S. Colla, *Computer-Aided Analysis of Decentralized Optimization Methods*, UCLouvain, 2024.


It gather codes used in different paper:
- S. Colla and J. M. Hendrickx, *Exploiting Agent Symmetries for Performance Analysis of Distribute Optimization Methods*, 2024. [PDF](https://arxiv.org/pdf/2403.11724)
- S. Colla and J. M. Hendrickx, *Automatic Performance Estimation for Decentralized Optimization*, in IEEE Transactions on Automatic Control, 2023. [PDF](https://arxiv.org/pdf/2203.05963)
- S. Colla and J. M. Hendrickx, *Automated Performance Estimation for Decentralized Optimization via Network Size Independent Problems*, in Proceedings of CDC 2022. [PDF](https://arxiv.org/pdf/2210.00695)

The codes allow computing the worst-case performance of decentralized optimization algorithm based on the Performance Estimation Problem (PEP) framework. 

### Requirements:
The codes are in Matlab, use the [YALMIP](https://yalmip.github.io/) toolbox and the [Mosek](https://www.mosek.com/) solver. Part of the codes also uses the [PESTO](https://github.com/PerformanceEstimation/Performance-Estimation-Toolbox) toolbox.

The folder `0_auxiliary functions` contains a few functions that are used in the codes. You should add this folder to the path, e.g., using the commands:
```
addpath("0_auxiliary functions");
savepath;
```

### Organisation
There is one folder for each decentralized algorithm considered (DGD, DIGing, EXTRA and Acc-DNGD). These folders contain Matlab functions to compute the worst-case performance of the algorithm in different settings. The different functions build on different PEP formulations that have been developed in the above references.

In addition, the folder `1_PESTO - script examples` contains simple scripts that rely on the PESTO toolbox and compute the worst-case performance of the algorithm. These scripts applies to a specific setting (class of functions, performance criterion, etc.) but can easily be adapted.
