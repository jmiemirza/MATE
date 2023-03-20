from copy import deepcopy
import torch.jit
import torch.optim as optim


def setup_tent(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model(args, model)  #  set only Batchnorm3d layers to trainable,   freeze all the other layers
    params, param_names = collect_params(model)  # collecting gamma and beta in all Batchnorm3d layers
    optimizer = setup_optimizer(args, params)  # todo hyperparameters are hard-coded above
    return model, optimizer


def setup_optimizer(tent_args, params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    return optim.Adam(params,
                      lr=tent_args.LR,
                      betas=(tent_args.BETA, 0.999),
                      weight_decay=tent_args.WD)



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model.module.classification_only(x, only_unmasked=False)  # (batch * n_views, 3, T, 224,224 )  -> (batch * n_views, n_class ) todo clip-level prediction
    loss = softmax_entropy(outputs).mean(0)   #   todo compute the entropy for all clip-level predictions   then take the averaga among all samples
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale gamma, bias is shift beta
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(args, model):
    """Configure model for use with tent."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.requires_grad_(True)
        m.track_running_stats = False # for original implementation this is False
        m.running_mean = None # for original implementation uncomment this
        m.running_var = None # for original implementation uncomment this

    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"

    has_bn = any([isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
