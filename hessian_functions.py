import torch
from torch.autograd.functional import vhp
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from grad_functions import grad_z, grad_z_group
from utils import make_functional, load_weights, calc_loss


def s_test_sample(
        model,
        top_model,
        x_test,
        y_test,
        train_loader,
        args,
        loss_func="cross_entropy",
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.float) for params in top_model.parameters()
    ]

    for i in range(args.r_average):
        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(
                train_loader.dataset, True, num_samples=args.recur_depth
            ),
            batch_size=args.hvp_batch_size,
            num_workers=4,
        )

        cur_estimate = s_test(
            x_test, y_test, model, top_model, i, hessian_loader, args, loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / args.scale) for old, cur in zip(inverse_hvp, cur_estimate)
            ]

    with torch.no_grad():
        inverse_hvp = [component / args.r_average for component in inverse_hvp]

    return inverse_hvp


def s_test_group_sample(
        model,
        top_model,
        val_loader,
        train_loader,
        args,
        loss_func="cross_entropy",
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.float) for params in top_model.parameters()
    ]

    for i in range(args.r_average):
        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(
                train_loader.dataset, True, num_samples=args.recur_depth
            ),
            batch_size=args.hvp_batch_size,
            num_workers=4,
        )

        cur_estimate = s_test_group(
            val_loader, model, top_model, i, hessian_loader, args, loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / args.scale) for old, cur in zip(inverse_hvp, cur_estimate)
            ]

    with torch.no_grad():
        inverse_hvp = [component / args.r_average for component in inverse_hvp]

    return inverse_hvp


def s_test(x_test, y_test, model, top_model, i, samples_loader, args, loss_func="cross_entropy"):

    v = grad_z(x_test, y_test, model, top_model, args, loss_func=loss_func)
    #     print("v: {}".format(v))
    h_estimate = v

    params, names = make_functional(top_model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    for i, (x_train, y_train) in enumerate(progress_bar):

        x_train, y_train = x_train.to(args.device), y_train.to(args.device)

        def f(*new_params):
            load_weights(top_model, names, new_params)
            model(x_train)
            out = top_model(model.embedding)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - args.damp) * _h_e - _hv / args.scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            if i % 100 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(top_model, names, params, as_params=True)

    return h_estimate


def s_test_group(val_loader, model, top_model, i, samples_loader, args,
                 loss_func="cross_entropy"):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
    Returns:
        h_estimate: list of torch tensors, s_test"""

    v = grad_z_group(val_loader, model, top_model, args)
    h_estimate = v

    params, names = make_functional(top_model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    for i, (x_train, y_train) in enumerate(progress_bar):

        x_train, y_train = x_train.to(args.device), y_train.to(args.device)

        def f(*new_params):
            load_weights(top_model, names, new_params)
            model(x_train)
            out = top_model(model.embedding)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - args.damp) * _h_e - _hv / args.scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            if i % 50 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(top_model, names, params, as_params=True)

    return h_estimate
