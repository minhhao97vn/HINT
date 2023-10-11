import copy
import json
import pickle
import sys

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from tqdm import tqdm

from dataset import CustomCIFAR10, CustomMNIST, CustomCIFAR10V2
from diff_data_augmentation import RandomTransform
from influence_functions import calc_influence_single_group_upweight, calc_influence_single_group_pert
from utils import set_attr
import os

mean_cifar10 = torch.tensor((0.4914, 0.4822, 0.4465))[:, None, None]
std_cifar10 = torch.tensor((0.2023, 0.1994, 0.2010))[:, None, None]


# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)


# poison_path = 'poison/mixed_poison_clean_29400_pgd_0_p1_0_p5_19600_DC_0.pt'


def display_progress(text, current_step, last_step, enabled=True, fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step - 1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def load_mnist(shuffle_train=True):
    torch.manual_seed(1)
    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    trainset_n, valset = torch.utils.data.random_split(trainset, [59000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset_n, batch_size=100,
                                              shuffle=shuffle_train, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=1)

    count = [0 for i in range(10)]
    for i in range(len(valset)):
        count[valset[i][1]] += 1
    print(count)

    return trainloader, valloader, testloader


def load_cifar10_targeted(shuffle_train=True, augmentation=True, args=None):
    torch.manual_seed(1)
    params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
    train_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    augmentation_transform = None

    if augmentation:
        print("Use augmentation")
        augmentation_transform = transforms.Compose([RandomTransform(**params, mode='bilinear')])
    else:
        print("No use augmentation")

    trainset = CustomCIFAR10V2(root='./data', train=True, split_train=True,
                               download=True, augmentation=augmentation_transform, transform=train_transform)

    valset = CustomCIFAR10(root='./data', train=True, split_train=False,
                           download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    if not args.benign:
        print(f"Adding poison from path: {args.poison_path}")

        # Since metapoison re-index the dataset, we need to split train-val again
        # Moreover, we only get 490 poisoned samples out of 500 poisoned samples downloaded from the author's source

        if 'metapoison' in args.poison_path:
            print("Setting metapoison ...")
            with open(args.poison_path, 'rb') as handle:
                poison_data = pickle.load(handle)

                train_indices = []
                val_indices = []
                for cl in range(10):
                    train_indices.append(np.array(range(5000 * cl, 5000 * cl + 4900)))
                    val_indices.append(np.array(range(5000 * cl + 4900, 5000 * (cl + 1))))
                train_indices = np.concatenate(train_indices, axis=0)
                val_indices = np.concatenate(val_indices, axis=0)

                trainset.data = np.uint8(poison_data['xtrain'])[train_indices]
                trainset.targets = poison_data['ytrain'][train_indices]
                trainset.original_data = copy.deepcopy(trainset.data)
                valset.data = np.uint8(poison_data['xtrain'])[val_indices]
                valset.targets = poison_data['ytrain'][val_indices]
                testset.data = np.uint8(poison_data['xvalid'])
                testset.targets = poison_data['yvalid']

                # target_img = test_transform(to_pil(np.uint8(poison_data['xtarget'][0])))
                target_class = poison_data['ytarget'][0]
                poisoned_label = poison_data['ytargetadv'][0]
                args.target_intended_class = poisoned_label
                poison_indices = np.array(range(4900 * poisoned_label, 4900 * poisoned_label + 500))
                args.target_sample_idx = 1000 * target_class + int(args.poison_path[-5])
                print("Target sample idx: ", args.target_sample_idx)
        else:
            with open(args.poison_path, 'rb') as handle:
                poison_data = pickle.load(handle)
                poison_indices = poison_data['poison_ids']
                poison_deltas = poison_data['poison_delta']
                if 'target_images' in poison_data.keys():
                    args.target_sample_idx = poison_data['target_images'][0][2]
                    args.target_intended_class = poison_data['poison_setup']['intended_class'][0]
            trainset.set_poison_delta(poison_indices.tolist(), poison_deltas)

        if 'bullseye' in args.poison_path or 'poisonfrogs' in args.poison_path:
            # Seed should be the same as seed used when pre-train the victim model such that training and transfer
            # sets are not overlapped
            np.random.seed(args.seed)
            transfer_set = poison_indices
            poison_class = poison_data['poison_setup']['poison_class']
            print(poison_class)
            for cl in range(10):
                if cl != poison_class:
                    class_ids = np.where(trainset.targets == cl)
                    transfer_set = np.concatenate([transfer_set, np.random.choice(class_ids[0], 490, replace=False)],
                                                  axis=0)
            trainset.set_subset(transfer_set)

    # Used when we pre-train victim model for transfer learning setting
    if args.scenario == 'scratch' and ('bullseye' in args.poison_path or 'poisonfrogs' in args.poison_path):
        print("===== Train victim model for transfer learning scenario =====")
        print(f"Loading poison deltas from path: {args.poison_path}")
        print("Setting training subset ...")
        with open(args.poison_path, 'rb') as handle:
            poison_data = pickle.load(handle)
            poison_indices = poison_data['poison_ids']
            np.random.seed(args.seed)
            transfer_set = poison_indices
            poison_class = poison_data['poison_setup']['poison_class']
            print(poison_class)
            for cl in range(10):
                if cl != poison_class:
                    class_ids = np.where(trainset.targets == cl)
                    transfer_set = np.concatenate([transfer_set, np.random.choice(class_ids[0], 490, replace=False)],
                                                  axis=0)
            remain = np.array([elem for elem in np.arange(49000) if elem not in transfer_set])
            print(transfer_set)
            print(len(np.unique(transfer_set)))
            print('remain size ', len(remain))
            trainset.set_subset(remain)
            print('train size ', len(trainset.data))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=shuffle_train, num_workers=4)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=4)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4)

    count = [0 for i in range(10)]
    for i in range(len(valset)):
        count[valset[i][1]] += 1
    print(count)

    return trainloader, valloader, testloader


def load_cifar10_untargeted(shuffle_train=True, augmentation=True, args=None):
    torch.manual_seed(1)
    params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
    train_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    augmentation_transform = None

    if augmentation:
        print("Use augmentation")
        augmentation_transform = transforms.Compose([RandomTransform(**params, mode='bilinear')])
    else:
        print("No use augmentation")

    trainset = CustomCIFAR10V2(root='./data', train=True, split_train=True,
                               download=True, augmentation=augmentation_transform, transform=train_transform)

    print(trainset)

    if not args.benign:
        print(f"Adding poison deltas from path: {args.poison_path}")

        # Use when poisoned data generated from DeepConfuse and Delusive attacks
        poison_data = torch.load(args.poison_path)
        trainset.update_poison_data_ver1(poison_data)

        # Use when poisoned data generated from torch attacks
        # with open(poison_path, 'rb') as handle:
        #     poison_data = pickle.load(handle)
        #     trainset.update_poison_data_ver2(poison_data)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=shuffle_train, num_workers=4)

    valset = CustomCIFAR10(root='./data', train=True, split_train=False,
                           download=True, transform=train_transform)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4)

    count = [0 for i in range(10)]
    for i in range(len(valset)):
        count[valset[i][1]] += 1
    print(count)

    return trainloader, valloader, testloader


def load_cifar10_subset_train(selected_indices, current_loader, shuffle_train=True):
    trainset = copy.deepcopy(current_loader.dataset)
    trainset.set_subset(selected_indices)
    trainset.augmentation = None

    return torch.utils.data.DataLoader(trainset, batch_size=128,
                                       shuffle=shuffle_train, num_workers=4)


def train_cifar10(train_loader, test_loader, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_schedule)

    best_acc = 0

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        progress_bar = tqdm(train_loader, desc=f"Base model, epoch: {epoch}", unit="batch")
        for i, (inputs, targets) in enumerate(progress_bar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(args.device), targets.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            progress_bar.set_postfix({"loss": loss.item()})

        scheduler.step()

        model.eval()
        # acc, _, _ = test(test_loader, model, print_result=(epoch % 5 == 0), args=args)
        acc, _, _ = test(test_loader, model, print_result=True, args=args)
        model.train()
        if acc > best_acc:
            best_acc = acc
    print('Finished Training')
    print("Best accuracy:", best_acc)


def get_healthy_noise(feature_influence):
    healthy_noise = []

    healthy_noise_max = -1
    healthy_noise_min = 1000

    for idx in range(len(feature_influence)):
        infl_map = feature_influence[idx]['infl'][0].cpu()
        healthy_noise.append(infl_map)

        if healthy_noise_max < infl_map.max():
            healthy_noise_max = infl_map.max()
        if healthy_noise_min > infl_map.min():
            healthy_noise_min = infl_map.min()

    print("Inf max:", healthy_noise_max)
    print("Inf min:", healthy_noise_min)

    return healthy_noise


def get_indices_to_add_HIN(model, top_model, train_loader_noshfl, val_loader, args):
    influences, s_test_vec = calc_influence_single_group_upweight(model, top_model, train_loader_noshfl, val_loader,
                                                                  args)
    # Get most positive and most negative influences
    sorted_indices = [i for i, x in sorted(enumerate(influences), key=lambda x: torch.abs(x[1]), reverse=True)]
    print('Choose from 25% most positive and 25% most negative')

    # Get most negative/positive influences
    # sorted_indices = [i for i, x in sorted(enumerate(influences), key=lambda x: x[1], reverse=True)]
    # print("Choose from 50% most positive")

    influences = np.array([item.cpu().detach().numpy() for item in influences])
    selected_indices = sorted_indices[:math.ceil(args.ratio * len(sorted_indices))]

    selected_influences = influences[selected_indices]
    return influences, selected_influences, selected_indices, s_test_vec


def update_top_model(model, top_model):
    fc_params = {}
    for name, param in model.named_parameters():
        if "linear" in name:
            fc_params[name] = param

    for name, param in fc_params.items():
        set_attr(top_model, name.split("."), param)
    return top_model


def train_cifar10_with_hin(train_loader, val_loader, test_loader, model, top_model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_schedule)

    selected_hin_indices = None

    best_result = {"best_acc": 0, "target_predicted": -1, "target_label": -1, "at_epoch": -1}
    test_acc = []
    target_predicteds = []

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        if epoch in args.hin_schedule:
            top_model = update_top_model(model, top_model)

            model.eval()
            top_model.eval()

            if selected_hin_indices is None:
                if args.ratio == 1.0:
                    selected_hin_indices = [i for i in range(len(train_loader.dataset.data))]
                    s_test_vec = None
                else:
                    train_loader_noshfl, _, _ = load_cifar10_untargeted(shuffle_train=False, augmentation=False,
                                                                      args=args)
                    all_influences, selected_hin_influences, selected_hin_indices, s_test_vec = get_indices_to_add_HIN(
                        model, top_model,
                        train_loader_noshfl,
                        val_loader, args)

                    to_save_file = {'selected_hin_indices': selected_hin_indices,
                                    "selected_hin_influences": selected_hin_influences,
                                    "all_influences": all_influences}

                    with open(f'selected_subset_cifar10_p00_pos25_neg25_at_epoch_{epoch}.pkl', 'wb') as handle:
                        pickle.dump(to_save_file, handle, pickle.HIGHEST_PROTOCOL)
            else:
                s_test_vec = None

            # s_test_vec = None
            # selected_hin_indices = train_loader.dataset.poison_indices

            sub_train_loader = load_cifar10_subset_train(selected_hin_indices, train_loader, shuffle_train=False)

            feature_influence = calc_influence_single_group_pert(model, top_model, sub_train_loader, val_loader,
                                                                 args, s_test_vec)

            # feature_influence = calc_influence_single_group_pert(model, top_model, sub_train_loader, val_loader,
            #                                                           args, s_test_vec)

            print("Updated data loader with HIN")
            healthy_noise = get_healthy_noise(feature_influence)

            if epoch >= 40:
                gamma = args.gamma / 10
            else:
                gamma = args.gamma

            # Choose to use sign or not?
            # train_loader.dataset.set_healthy_noise(selected_hin_indices, healthy_noise, gamma)
            if args.sign:
                train_loader.dataset.set_healthy_noise(selected_hin_indices, healthy_noise, gamma, args.eps)
            else:
                train_loader.dataset.set_healthy_noise_no_sign(selected_hin_indices, healthy_noise, gamma, args.eps,
                                                               epoch)
            model.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for i, (inputs, targets) in enumerate(progress_bar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(args.device), targets.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

        model.eval()
        # total_acc, target_sample_predicted, target_sample_label = test(test_loader, model, args,
        #                                                                print_result=(epoch % 5 == 0))
        total_acc, target_sample_predicted, target_sample_label = test(test_loader, model, args,
                                                                       print_result=True)
        model.train()

        test_acc.append(float(total_acc))
        target_predicteds.append(target_sample_predicted)
        if total_acc > best_result["best_acc"]:
            best_result["best_acc"] = float(total_acc)
            best_result["at_epoch"] = epoch
            best_result["target_predicted"] = target_sample_predicted
            best_result["target_label"] = target_sample_label

        scheduler.step()
    print('Finished Training')
    model.eval()
    test(test_loader, model, args, print_result=True)
    model.train()

    if args.target_sample_idx is not None:
        best_result['test_acc_epochs'] = test_acc
        best_result['target_predicted_epochs'] = target_predicteds
    temp_args = copy.deepcopy(args)
    temp_args.device = None
    best_result['setup'] = vars(temp_args)
    print(best_result)
    print(test_acc)
    # j_obj = json.dumps(best_result)
    # with open(f"output/best_result_{args.seed}_poison_{[args.poison_path[7:14]]}.json", 'w') as f:
    #     f.write(j_obj)


def load_mnist_untargeted(shuffle_train=True, args=None):
    torch.manual_seed(1)

    train_test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = CustomMNIST(root='./data', train=True, split_train=True,
                           download=True)

    if not args.benign:
        print(f"Adding poison deltas from path: {args.poison_path}")
        with open(args.poison_path, 'rb') as handle:
            poison_data = torch.load(handle)
            trainset.update_poison_data_ver1(poison_data)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=shuffle_train, num_workers=4)

    valset = CustomMNIST(root='./data', train=True, split_train=False,
                         download=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=train_test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4)

    count = [0 for i in range(10)]
    for i in range(len(valset)):
        count[valset[i][1]] += 1
    print(count)

    return trainloader, valloader, testloader


def load_mnist_subset_train(selected_indices, current_loader, shuffle_train=True):
    trainset = copy.deepcopy(current_loader.dataset)
    trainset.set_subset(selected_indices)
    trainset.augmentation = None

    return torch.utils.data.DataLoader(trainset, batch_size=128,
                                       shuffle=shuffle_train, num_workers=4)


def train_mnist_with_hin(train_loader, val_loader, test_loader, model, top_model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    selected_hin_indices = None

    best_result = {"best_acc": 0, "target_predicted": -1, "target_label": -1, "at_epoch": -1}
    test_acc = []
    target_predicteds = []

    torch.manual_seed(args.seed)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        if epoch in args.hin_schedule:
            top_model = update_top_model(model, top_model)

            model.eval()
            top_model.eval()

            if selected_hin_indices is None:
                if args.ratio == 1.0:
                    selected_hin_indices = [i for i in range(len(train_loader.dataset.data))]
                    s_test_vec = None
                else:
                    train_loader_noshfl, _, _ = load_mnist_untargeted(shuffle_train=False, args=args)
                    all_influences, selected_hin_influences, selected_hin_indices, s_test_vec = get_indices_to_add_HIN(
                        model, top_model,
                        train_loader_noshfl,
                        val_loader, args)

                    to_save_file = {'selected_hin_indices': selected_hin_indices, "selected_hin_influences": selected_hin_influences}

                    with open('selected_subset_mnist_p06_pos0_neg50.pkl', 'wb') as handle:
                        pickle.dump(to_save_file, handle, pickle.HIGHEST_PROTOCOL)
            else:
                s_test_vec = None

            sub_train_loader = load_mnist_subset_train(selected_hin_indices, train_loader, shuffle_train=False)

            feature_influence = calc_influence_single_group_pert(model, top_model, sub_train_loader, val_loader,
                                                                 args, s_test_vec)
            print("Updated data loader with HIN")
            healthy_noise = get_healthy_noise(feature_influence)
            if epoch >= 40:
                gamma = args.gamma / 10
            else:
                gamma = args.gamma
            train_loader.dataset.set_healthy_noise_no_sign(selected_hin_indices, healthy_noise, gamma, args.eps, epoch)
            model.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for i, (inputs, targets) in enumerate(progress_bar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(args.device), targets.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

        model.eval()
        total_acc, target_sample_predicted, target_sample_label = test(test_loader, model, args,
                                                                       print_result=True)
        model.train()

        test_acc.append(float(total_acc))
        target_predicteds.append(target_sample_predicted)
        if total_acc > best_result["best_acc"]:
            best_result["best_acc"] = float(total_acc)
            best_result["at_epoch"] = epoch
            best_result["target_predicted"] = target_sample_predicted
            best_result["target_label"] = target_sample_label

    print('Finished Training')
    model.eval()
    test(test_loader, model, args,
         print_result=True)
    model.train()

    if args.target_sample_idx is not None:
        best_result['test_acc_epochs'] = test_acc
        best_result['target_predicted_epochs'] = target_predicteds
    temp_args = copy.deepcopy(args)
    temp_args.device = None
    best_result['setup'] = vars(temp_args)
    print(best_result)
    j_obj = json.dumps(best_result)
    # with open(f"output/best_result_{args.seed}_poison_{args.poison_path[7:14]}.json", 'w') as f:
    #     f.write(j_obj)


def train_mnist(train_loader, test_loader, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    best_acc = 0
    torch.manual_seed(args.seed)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        progress_bar = tqdm(train_loader, desc=f"Base model, epoch: {epoch}", unit="batch")
        for i, (inputs, targets) in enumerate(progress_bar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(args.device), targets.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            progress_bar.set_postfix({"loss": loss.item()})

        model.eval()
        # acc, _, _ = test(test_loader, model, print_result=(epoch % 5 == 0), args=args)
        acc, _, _ = test(test_loader, model, print_result=True, args=args)
        model.train()
        if acc > best_acc:
            best_acc = acc
    print('Finished Training')
    print("Best accuracy:", best_acc)


def test_mnist(testloader, net, device):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
            c = (pred == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the %d test images: %d %%' % (len(testloader.dataset),
                                                                    100 * correct / total))
    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def test(test_loader, net, args, print_result=False, prefix=None):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    if args.target_sample_idx is not None:
        target_sample_idx = args.target_sample_idx
        target_sample = test_loader.dataset.data[target_sample_idx]
        target_sample = test_loader.dataset.transform(target_sample).to(args.device)
        target_sample = target_sample[None, :, :, :]
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
            c = (pred == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        if args.target_sample_idx is not None:
            target_sample_prediction = torch.max(net(target_sample), 1)[1].cpu().detach().numpy()[0]
            target_sample_pred_prob = F.softmax(net(target_sample))[0].cpu().detach().numpy()
            target_sample_label = test_loader.dataset.targets[target_sample_idx]

    total_acc = 100 * correct / total
    if print_result:
        if args.target_sample_idx is not None:
            print(f"target sample prediction: {target_sample_prediction}, prob: {target_sample_pred_prob}")
            print("target intended class: ", args.target_intended_class)
            print("target sample label: ", target_sample_label)
        print('Accuracy of the network on the {} test images: {:.2f} %%'.format(len(test_loader.dataset), total_acc))

    if args.save_result and prefix is not None:
        final_result = {"total_acc": total_acc}
        if args.target_sample_idx is not None:
            final_result['target_sample_prediction'] = int(target_sample_prediction)
            final_result['target_sample_label'] = int(target_sample_label)
            final_result['target_intended_class'] = int(args.target_intended_class)
        classes = ('0', '1', '2', '3',
                   '4', '5', '6', '7', '8', '9')
        for i in range(10):
            # print('Accuracy of %5s : %2d %%' % (
            #     classes[i], 100 * class_correct[i] / class_total[i]))
            final_result[classes[i]] = 100 * class_correct[i] / class_total[i]
        j_obj = json.dumps(final_result)
        with open(f"output/{prefix}_{args.seed}_ratio_{args.ratio}.json", 'w') as f:
            f.write(j_obj)
    if args.target_sample_idx is not None:
        return total_acc, target_sample_prediction, target_sample_label
    else:
        return total_acc, None, None
