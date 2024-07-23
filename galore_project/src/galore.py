import math

import torch
from torch.optim import Optimizer


def _project_gradient(grad, P):
    if P.shape[0] <= P.shape[1]:
        return P.t() @ grad
    else:
        return grad @ P


def _project_back(update, P):
    if P.shape[0] <= P.shape[1]:
        return P @ update
    else:
        return update @ P.t()


class GaLore(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 rank=4, update_proj_gap=200, alpha=0.25, use_cuda=torch.cuda.is_available()):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, update_proj_gap=update_proj_gap, alpha=alpha)
        super(GaLore, self).__init__(params, defaults)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GaLore does not support sparse gradients')

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['P'] = None

                self._update_step(p, grad, group, state)

        return loss

    def _update_step(self, p, grad, group, state):
        state['step'] += 1
        beta1, beta2 = group['betas']
        rank = group['rank']
        update_proj_gap = group['update_proj_gap']
        alpha = group['alpha']

        # Update projection matrix if needed
        if state['step'] % update_proj_gap == 0 or state['P'] is None:
            state['P'] = self._compute_projection_matrix(grad, rank)

        # Project gradient
        proj_grad = _project_gradient(grad, state['P'])

        # Apply weight decay
        if group['weight_decay'] != 0:
            proj_grad = proj_grad.add(p, alpha=group['weight_decay'])

        # Update moments
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        exp_avg.mul_(beta1).add_(proj_grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(proj_grad, proj_grad, value=1 - beta2)

        # Compute bias correction terms
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Compute adaptive learning rate
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        # Compute the Adam update
        denom = exp_avg_sq.sqrt().add_(group['eps'])
        update = exp_avg / denom

        # Project update back to original space
        update = _project_back(update, state['P'])

        # Apply update
        p.add_(update, alpha=-step_size * alpha)

    def _compute_projection_matrix(self, grad, rank):
        m, n = grad.shape
        if m <= n:
            U, _, _ = torch.svd_lowrank(grad, q=rank)
            return U.to(self.device)
        else:
            _, _, V = torch.svd_lowrank(grad.t(), q=rank)
            return V.to(self.device)
