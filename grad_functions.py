import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable

from utils import calc_loss


def grad_z(x, y, model, top_model, args, loss_func="cross_entropy"):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        x: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        y: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()

    # initialize
    x, y = x.to(args.device), y.to(args.device)

    model(x)

    prediction = top_model(model.embedding)

    loss = calc_loss(prediction, y, loss_func=loss_func)

    # Compute sum of gradients from model parameters to loss
    return grad(loss, top_model.parameters())


def grad_z_group(val_loader, model, top_model, args):
    model.eval()
    top_model.eval()

    total_loss = None
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
        model(x_val)
        prediction = top_model(model.embedding)
        if total_loss is None:
            total_loss = F.cross_entropy(prediction, y_val, reduction='none')
        else:
            total_loss = torch.cat((total_loss, F.cross_entropy(prediction, y_val, reduction='none')), 0)

    loss = total_loss.mean()

    # Compute sum of gradients from model parameters to loss
    return grad(loss, top_model.parameters())


def double_grad_wrt_input(x, y, grad_test_hessian, model, args, top_params, loss_func="cross_entropy"):
    model.eval()

    # initialize
    x, y = x.to(args.device), y.to(args.device)

    var_x = Variable(x.data, requires_grad=True)

    prediction = model(var_x)

    loss = calc_loss(prediction, y, loss_func=loss_func)

    # Compute sum of gradients from model parameters to loss
    grad_theta = grad(loss, tuple(top_params), retain_graph=True, create_graph=True)

    elementwise_products = torch.zeros(1).to(args.device)
    for grad_elem, ih_elem in zip(grad_theta, grad_test_hessian):
        elementwise_products += torch.sum(grad_elem * ih_elem.detach())

    grad_input = grad(elementwise_products, var_x)

    return grad_input


def double_grad_wrt_input_fast(samples, targets, grad_test_hessian, model, args, top_params, loss_func="cross_entropy"):
    model.eval()

    # initialize
    samples, targets = samples.to(args.device), targets.to(args.device)

    samples = torch.stack([torch.tensor(samples[i], requires_grad=True) for i in range(len(samples))], dim=0)

    # top_params_all = torch.stack([torch.tensor(top_params) for i in range(len(samples))], dim=0)
    # print(len(top_params_all))
    # print(top_params_all.shape)
    # prediction_all = model(samples)
    # loss_all = F.cross_entropy(prediction_all, targets, reduction='none')


    influences = []

    for i in range(len(samples)):
        sample = samples[i][None, :]

        prediction = model(sample)

        # Compute sum of gradients from model parameters to loss
        loss = F.cross_entropy(prediction, targets[i].reshape(1))

        grad_theta = grad(loss, top_params, retain_graph=True, create_graph=True)

        elementwise_products = torch.zeros(1).to(args.device)
        for grad_elem, ih_elem in zip(grad_theta, grad_test_hessian):
            elementwise_products += torch.sum(grad_elem * ih_elem.detach())

        grad_out = grad(elementwise_products, samples)

        influences.append(grad_out[0])

    return influences
