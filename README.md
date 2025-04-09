# Deep Variational Information Bottleneck
<br>

### Overview
More Modern Adaptation of the Pytorch implementation of Deep Variational Information Bottleneck([paper], [original code])

Original Pytorch Implementation: ([https://github.com/1Konny/VIB-pytorch])

![ELBO](misc/ELBO.PNG)
![monte_carlo](misc/monte_carlo.PNG)
<br>

### Setup
1. Download Mini Conda: https://www.anaconda.com/docs/getting-started/miniconda/main
2. Create and activate Conda Environment
```
conda create -n myenv python=3.11
conda activate myenv
```
3. Install the required packages:
```
pip install -r requirements.txt
```
<br>

### Usage
1. train
```
python main.py --mode train --beta 1e-3 --tensorboard True --env_name [NAME]
```
2. TensorBoard
```
tensorboard --logdir=summary/[NAME]/
```
2. test
```
python main.py --mode test --env_name [NAME] --load_ckpt best_acc.tar
```
<br>

### References
1. Deep Learning and the Information Bottleneck Principle, Tishby et al.
2. Deep Variational Information Bottleneck, Alemi et al.
3. Tensorflow Demo : https://github.com/alexalemi/vib_demo

[paper]: http://arxiv.org/abs/1612.00410
[original code]: https://github.com/alexalemi/vib_demo
