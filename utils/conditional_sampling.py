
import torch
import torch.nn.functional as F


def get_cfg_conditional_distribution(lambda_guidance, denoiser, cond, x_t, a_t, mask, max_num_nodes):
    cond_off = torch.eye(4)[-1].view(1, 1, -1).repeat(*x_t.shape[:2], 1).to(x_t.device)
    x = torch.cat((x_t, cond_off), dim=-1)
    log_p_x_uncond, log_p_a_uncond = denoiser(x, a_t, mask.bool())

    cond = cond.repeat(max_num_nodes, 1).reshape(*x_t.shape[:2], -1) * mask.unsqueeze(-1)
    cond_indicator = torch.zeros(*x_t.shape[:2], 1, device=x_t.device)
    x = torch.cat((x_t, cond, cond_indicator), dim=-1)
    log_p_x_cond, log_p_a_cond = denoiser(x, a_t, mask.bool())

    log_p_a = lambda_guidance * log_p_a_cond + (1 - lambda_guidance) * log_p_a_uncond
    log_p_x = lambda_guidance * log_p_x_cond + (1 - lambda_guidance) * log_p_x_uncond
    return log_p_x.softmax(-1), log_p_a.softmax(-1)

def get_gradiant(model, X, A, mask, condition):
    grad = compute_input_gradient(model, X, A, mask, condition)
    return grad

def compute_input_gradient(model, X, A, mask, condition):
    """
    Computes the gradient of the model's output with respect to the input.

    Args:
        model (torch.nn.Module): The neural network model.
        input_tensor (torch.Tensor): The input tensor for which we compute the gradient.
        target_output (torch.Tensor, optional): If provided, the function computes gradients of the loss
                                                with respect to the input. Otherwise, it computes gradients
                                                of the sum of outputs.

    Returns:
        torch.Tensor: The computed gradients.
    """
    # Ensure the input requires gradient
    X, A = X.clone().detach(), A.clone().detach()
    X.requires_grad_(True), A.requires_grad_(True)
    with torch.enable_grad():
        # Forward pass
        output, _ = model(X, A, mask.bool())
        # If target_output is given, compute loss-based gradients
        loss = F.mse_loss(output, condition)
        loss.backward()

    return X.grad, A.grad
