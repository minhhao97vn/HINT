import logging
import multiprocessing
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Subset


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = '%(asctime)s: %(message)s'
    if filename:
        logging.basicConfig(handlers=[logging.FileHandler(filename),
                                      logging.StreamHandler(sys.stdout)],
                            level=log_lvl,
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl,
                            format=log_format)


def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'dataset': 'cifar',
        'seed': 42,
        'gpu': 0,
        'num_classes': 10,
        'test_sample_num': 1,
        'test_start_index': 500,
        'recursion_depth': 10,
        'r_averaging': 10,
        'batch_size': 128,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
        'model_name': 'cifar_resnet18_200eps.pth',
        'load_model_name': 'cifar_resnet18_200eps.pth'
    }

    return config


def log_clip(x):
    return torch.log(torch.clamp(x, 1e-10, None))





def calc_loss(logits, labels, loss_func="cross_entropy"):
    """Calculates the loss
    Arguments:
        logits: torch tensor, input with size (minibatch, nr_of_classes)
        labels: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
        loss_func: str, specify loss function name
    Returns:
        loss: scalar, the loss"""

    if loss_func == "cross_entropy":
        if logits.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.float))
        else:
            loss = F.cross_entropy(logits, labels)
    elif loss_func == "mean":
        loss = torch.mean(logits)
    else:
        raise ValueError("{} is not a valid value for loss_func".format(loss_func))

    return loss


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))


def loader_subdata(sub_data, train=False, batch_size=128):
    if train:
        dataloader = torch.utils.data.DataLoader(sub_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=max(1, multiprocessing.cpu_count() - 1))
    else:
        dataloader = torch.utils.data.DataLoader(sub_data, batch_size=batch_size, shuffle=False,
                                                 num_workers=max(1, multiprocessing.cpu_count() - 1))
    return dataloader


def create_subloader(loader, indices, batch_size, is_train):
    sub_data = torch.utils.data.Subset(loader.dataset, indices)
    sub_loader = loader_subdata(sub_data, train=is_train, batch_size=batch_size)
    return sub_loader


def calc_test_loss(model, test_loader, args):
    total_loss = None
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            if total_loss is None:
                total_loss = F.cross_entropy(outputs, targets, reduction='none')
            else:
                total_loss = torch.cat((total_loss, F.cross_entropy(outputs, targets, reduction='none')), 0)
    return total_loss.mean().cpu().detach().numpy()


def predict_one_sample(test_sample, net, device):
    with torch.no_grad():
        image = test_sample.to(device)
        output = net(image)
        return torch.argmax(output)
