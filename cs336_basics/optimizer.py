from collections.abc import Callable, Iterable
from typing import Dict, Optional, Tuple, Any
from jaxtyping import Float, Bool, Int
import torch
from torch import Tensor
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, Tensor]], 
                 lr: Float=1e-3,
                 betas: Tuple[Float, Float] = (0.9, 0.999),
                 eps: Float=1e-8,
                 weight_decay: Float=1e-2
                 ) -> None:
        defaults: Dict[str, Any] = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Tensor reference
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # int, not reference
                state['t'] += 1
                t = state['t']
                
                # compute first and second moment estimate @reference
                grad = p.grad.data
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                # exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1-beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # compute ajusted lr for iteration t
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                # update parameters
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # apply weight decay
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
        return loss

                
def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif it <=cosine_cycle_iters:
        return min_learning_rate + (1+math.cos(math.pi *(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))* (max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate

def test_opt():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

if __name__ == '__main__':
    test_opt()
