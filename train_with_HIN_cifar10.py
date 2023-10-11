import argparse
import warnings

import matplotlib.pyplot as plt
import torch

from models import FCNet, ResNet18
from train_utils import train_cifar10_with_hin, test, load_cifar10_untargeted, load_cifar10_targeted
from utils import set_attr
import os
import torch.nn as nn

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training scheme
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_schedule", type=str, default="30,50,70")
    parser.add_argument("--poison_path", type=str, default="")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--augmentation", type=bool, default=True)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--benign", dest='benign', action='store_true')
    parser.add_argument("--no_benign", dest='benign', action='store_false')
    parser.add_argument("--scenario", type=str, default="scratch", choices=['scratch', 'transfer'])
    parser.add_argument("--pretrained_model", type=str,
                        default="ResNet18_CIFAR10_80eps_subset_bullseye_trial_11_s112341.pth")
    parser.add_argument("--save_model", dest='save_model', action='store_true')
    parser.set_defaults(benign=True)
    parser.set_defaults(save_model=False)
    # GPU
    parser.add_argument("--gpu_id", type=int, default=7)

    # HIN scheme
    parser.add_argument("--hin_schedule", type=str, default="40,60")
    parser.add_argument("--damp", default=0.01, type=int)
    parser.add_argument("--scale", default=50, type=int)
    parser.add_argument("--recur_depth", default=49000, type=int)
    parser.add_argument("--r_average", default=1, type=int)
    parser.add_argument("--hvp_batch_size", default=50, type=int)
    parser.add_argument("--gamma", default=0.01, type=float)
    parser.add_argument("--ratio", default=0.5, type=float)
    parser.add_argument("--eps", default=16, type=int)
    parser.add_argument("--sign", dest='sign', action='store_true')
    parser.add_argument("--no_sign", dest='sign', action='store_false')
    parser.set_defaults(sign=False)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.device = torch.device('cuda', args.gpu_id)
    plt.rcParams['figure.dpi'] = 300

    args.lr_schedule = [int(item) for item in args.lr_schedule.split(",")]
    args.hin_schedule = [int(item) for item in args.hin_schedule.split(",")]
    args.target_sample_idx = None
    args.target_intended_class = None
    print(args)

    torch.manual_seed(args.seed)

    # prepare ResNet model to train w
    train_loader, val_loader, test_loader = load_cifar10_targeted(shuffle_train=True, augmentation=True, args=args)
    torch_model = ResNet18()

    if args.scenario == 'transfer':
        print(f"Loading pretrained model: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model)
        torch_model.load_state_dict(checkpoint)

        print("Freezing feature extractor ...")
        for param in torch_model.parameters():
            param.requires_grad = False

        print("Re-initializing linear layer ...")
        num_ftrs = torch_model.linear.in_features
        num_classes = 10
        torch_model.linear = nn.Linear(num_ftrs, num_classes)
        print("Finish setting up transfer scenario")
    else:
        print("Finish setting up from-scratch scenario")

    torch_model.to(args.device)


    top_model = FCNet(input_size=torch_model.linear.in_features, output_size=torch_model.linear.out_features)

    fc_params = {}
    for name, param in torch_model.named_parameters():
        if "linear" in name:
            fc_params[name] = param

    for name, param in fc_params.items():
        set_attr(top_model, name.split("."), param)
    top_model.to(args.device)

    # Train model with HIN
    train_cifar10_with_hin(train_loader, val_loader, test_loader, torch_model, top_model, args)
    torch_model.eval()
    test(test_loader, torch_model, args, print_result=True, prefix="with_hin")
    torch_model.train()

    if args.save_model:
        print("Saving model ...")
        torch.save(torch_model.state_dict(),
                   f"HINT_RN18_CIFAR10_{args.epochs}eps_{args.poison_path[7:-4]}_s{args.seed}.pth")
        print(f"Model saved to HINT_RN18_CIFAR10_{args.epochs}eps_{args.poison_path[7:-4]}_s{args.seed}.pth")
