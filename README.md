 # Comparing & evaluating uncertainty estimation
 ## deep learning point estimate models vs bayesian deep learning density estimators

 - Bayesian deep learning methods
   - MC-Dropout
   - SWAG
   - DPN
   - JEM

- Datasets

  | Train/In Distribution | Test/Out Distribution |
  | --------------------- | --------------------- |
  | CIFAR10               | {CIFAR100, SVHN, LSUN}|
  | FashionMNIST          | {CIFAR100, CIFAR10, LSUN}|
  | CIFAR100              | {CIFAR10, SVHN, LSUN}|
  | SVHN                  | {CIFAR100, CIFAR10, LSUN}|
- Results
![](./imgs/results_table.png)
---
- <u>Code Attribution</u>
  - [Stochastic Weight Averaging of Gaussian](https://github.com/wjmaddox/swa_gaussian)
  - [Dirichlet Prior Networks](https://github.com/KaosEngineer/PriorNetworks)
  - [Joint Energy Models](https://github.com/wgrathwohl/JEM)
