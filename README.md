 # Comparing & evaluating uncertainty estimation
 ## deep learning point estimate models vs bayesian deep learning

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

- Pretrained Models

  | Pretrained WideResNet28x10 on CIFAR10         |
  | --------------------- | --------------------- |
  | DNN                   | {CIFAR100, SVHN, LSUN}|
  | MC-Dropout            | {CIFAR100, CIFAR10, LSUN}|
  | SWAG                  | {CIFAR10, SVHN, LSUN}|
  | DPN                   | {CIFAR100, CIFAR10, LSUN}|
  | [JEM](https://www.google.com) |

- Results
![](./imgs/results_table.png)
---
- <u>Code Attribution</u>
  - [Stochastic Weight Averaging of Gaussian](https://github.com/wjmaddox/swa_gaussian)
  - [Dirichlet Prior Networks](https://github.com/KaosEngineer/PriorNetworks)
  - [Joint Energy Models](https://github.com/wgrathwohl/JEM)
