import datetime
import logging

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from grad_functions import grad_z, double_grad_wrt_input, double_grad_wrt_input_fast
from hessian_functions import s_test_sample, s_test_group_sample

plt.rcParams['figure.dpi'] = 300


# Need update
def calc_influence_single_upweight(
        model,
        top_model,
        train_loader,
        test_loader,
        test_id_num,
        args,
        s_test_vec=None,
        time_logging=False,
        loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence_functions: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = s_test_sample(
            model,
            top_model,
            z_test,
            t_test,
            train_loader,
            args,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        if time_logging:
            time_a = datetime.datetime.now()

        grad_z_vec = grad_z(z, t, model, top_model, args)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                    -sum(
                        [
                            ####################
                            # TODO: potential bottle neck, takes 17% execution time
                            # torch.sum(k * j).data.cpu().numpy()
                            ####################
                            torch.sum(k * j).data
                            for k, j in zip(grad_z_vec, s_test_vec)
                        ]
                    )
                    / train_dataset_size
            )

        influences.append(tmp_influence)

    return influences, test_id_num


def calc_influence_single_group_upweight(
        model,
        top_model,
        train_loader,
        val_loader,
        args,
        s_test_vec=None,
        time_logging=False,
        loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence_functions: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided

    print("damp: {}, scale: {}".format(args.damp, args.scale))

    if s_test_vec is None:
        s_test_vec = s_test_group_sample(
            model,
            top_model,
            val_loader,
            train_loader,
            args,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        if time_logging:
            time_a = datetime.datetime.now()

        grad_z_vec = grad_z(z, t, model, top_model, args)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                    -sum([torch.sum(k * j).data for k, j in zip(grad_z_vec, s_test_vec)]) / train_dataset_size
            )

        influences.append(-tmp_influence)

    return influences, s_test_vec


def calc_influence_single_pert(
        model,
        top_model,
        train_loader,
        test_loader,
        test_id_num,
        args,
        s_test_vec=None,
        time_logging=False,
        loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence_functions: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""

    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = s_test_sample(
            model,
            top_model,
            z_test,
            t_test,
            train_loader,
            args,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    feature_influences = []

    top_params = []
    for name, param in model.named_parameters():
        if "linear" in name:
            top_params.append(param)

    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        if time_logging:
            time_a = datetime.datetime.now()

        influence_pert = double_grad_wrt_input(z, t, s_test_vec, model, args, top_params)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                    -sum(
                        [
                            ####################
                            # TODO: potential bottle neck, takes 17% execution time
                            # torch.sum(k * j).data.cpu().numpy()
                            ####################
                            torch.sum(k).data
                            for k in influence_pert
                        ]
                    )
                    / train_dataset_size
            )

        influences.append(tmp_influence)
        feature_influences.append({'img': z, 'infl': influence_pert})

    return influences, test_id_num, feature_influences


def calc_influence_single_group_pert(
        model,
        top_model,
        train_loader,
        val_loader,
        args,
        s_test_vec=None,
        time_logging=False,
        loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence_functions: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""

    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        s_test_vec = s_test_group_sample(
            model,
            top_model,
            val_loader,
            train_loader,
            args,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    feature_influence = []

    top_params = []
    for name, param in model.named_parameters():
        if "linear" in name:
            top_params.append(param)

    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        # print('img in infl func max ', z.max())
        # print('img in infl func max ', z.min())

        if time_logging:
            time_a = datetime.datetime.now()

        influence_pert = double_grad_wrt_input(z, t, s_test_vec, model, args, top_params)
        # influence_pert = [-item / train_dataset_size for item in influence_pert]
        influence_pert = [-item for item in influence_pert]

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )

        feature_influence.append({'img': z, 'infl': influence_pert})

    return feature_influence


def calc_influence_single_group_pert_fast(
        model,
        top_model,
        train_loader,
        val_loader,
        args,
        s_test_vec=None,
        time_logging=False,
        loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence_functions: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""

    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        s_test_vec = s_test_group_sample(
            model,
            top_model,
            val_loader,
            train_loader,
            args,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    feature_influence = []

    top_params = []
    for name, param in model.named_parameters():
        if "linear" in name:
            top_params.append(param)

    top_params = tuple(top_params)

    for data in tqdm(train_loader):
        samples, targets = data[0], data[1]

        if time_logging:
            time_a = datetime.datetime.now()

        influence_pert = double_grad_wrt_input_fast(samples, targets, s_test_vec, model, args, top_params)
        influence_pert = [-item / train_dataset_size for item in influence_pert]

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )


        # Need to concatenate here
        for idx in range(len(samples)):
            feature_influence.append({'img': samples[idx], 'infl': influence_pert[idx]})

    return feature_influence