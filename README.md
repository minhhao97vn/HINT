# Source code for the paper: ["HINT: Healthy Influential Noise-based Training to Defend against Data Poisoning Attacks"](https://arxiv.org/abs/2309.08549)

## Code base
Implementation uses Python 3.9 version with Pytorch. To install dependencies for this source code:
```
pip install -r requirements.txt
```
This source code contains python implementation for HINT method as long as pretrained models and poisoned data.

 - Pretrained models used for transfer learning scenarios are in ```saved_models``` folder.
 - Poisoned data files are in ```poison``` folder.


## Running experiments
### CIFAR-10
Use ```train_with_HIN_cifar10.py``` to run experiments with CIFAR-10 dataset.

For examples: Use the following command to train ResNet-18 with HINT on from-scratch scenario. The attack is Meta Poison.
```
python train_with_HIN_cifar10.py --seed 311113 --gpu_id 3 --no_benign --scenario "scratch" --gamma 0.1 --no_sign --hin_schedule "5,15,40" --poison_path 'poison/CIFAR10/metapoison-dataset-resnet-frogplane-2.pkl'
```
On transfer learning scenario, use the following command:
```
python train_with_HIN_cifar10.py --seed 211112 --gpu_id 3 --no_benign --scenario "transfer" --gamma 0.1 --hin_schedule "5,15,40" --poison_path 'poison/CIFAR10/poisonfrogs_trial_3_poisons_packed_2023-05-01.pkl' --pretrained_model 'saved_models/ResNet18_CIFAR10_80eps_subset_poisonfrogs_trial_s211113.pth'
```

### MNIST
Use ```train_with_HIN_mnist.py``` to run experiments with MNIST dataset.

For examples: Use the following command to train CNN with HINT on from-scratch scenario. The attack ratio \rho is 0.6.
```
python train_with_HIN_mnist.py --no_benign --gamma 0.1 --hin_schedule "9" --poison_path 'poison/MNIST/mnist_eps_0.3_mixed_poison_clean_23600_pgd_8850_p1_8850_p5_8850_DC_8850.pt' --ratio 0.5 --seed 611116 --gpu_id 1
```

## For other defense baselines
### FRIENDS 
We use and follow the implementation from [FRIENDS](https://github.com/tianyu139/friendly-noise)
### ATDA 
We use and follow the implementation from [ATDA](https://github.com/TLMichael/Delusive-Adversary)
### EPIC
We use and follow the implementation from [EPIC](https://github.com/YuYang0901/EPIC)
