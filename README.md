 # Comparing & evaluating uncertainty estimation
 ## deep learning point estimate models vs bayesian deep learning density estimators

 - Point estimate deep learning models
   - alexnet
   - resnet50
   - densenet128
   - resnet18 (test accuracy 82.97%)
   - densenet_cifar (test accuracy 89.53%)
   - googlenet (test accuracy 92.5%)

 - Bayesian deep learning density estimators
   - TBA
   - TBA
   - TBA

- Datasets

  | Train/In Distribution | Test/Out Distribution | Test/Out Distribution |
  | --------------------- | --------------------- | --------------------- |
  | MNIST                 | Not MNIST             | MNIST (hold-out 5 classes) |
  | FashionMNIST          | MNIST                 | FashionMNIST (hold-out 5 classes) |
  | CIFAR                 | SVHN                  | CIFAR (hold-out 5 classes) |

- Results
![](./imgs/results_table.png)
