# PyTorch Implementation of  iCaRL



A PyTorch Implementation of [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725).



## requirement

python 3.10.16

torch pytorch     2.2.2  py3.10_cuda11.8_cudnn8.7.0_0  pytorch
torchaudio        2.2.2   py310_cu118    pytorch
torchvision       0.17.2   py310_cu118    pytorch

numpy

PIL



## run

```shell
python -u main.py
```





# Result

Resnet18+CIFAR100



| incremental step    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9|
| ------------------- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| iCaRL test accuracy | 83.8|77.81|74.332|71.244|68.252|64.788|61.756|58.588|56.546|54.108|
