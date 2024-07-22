import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Tuple, Optional
import math
import warnings

class GaLore(Optimizer):
    """
    GaLore (Gradient Low-Rank Projection) optimizer.
    
    This optimizer implements the GaLore algorithm for memory-efficient training of
    large language models as described in the paper "GaLore: Memory-Efficient LLM 
    Training by Gradient Low-Rank Projection".
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages
            of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        rank (int): rank for low-rank projection
        update_proj_gap (int): number of steps between projection matrix updates
        alpha (float): scale factor for gradient updates
        use_cuda (bool): whether to use CUDA for computations (default: True if available)
        ema_decay (float): decay factor for exponential moving average of projection matrices (default: 0.99)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 rank=4, update_proj_gap=200, alpha=0.25, use_cuda=torch.cuda.is_available(),
                 ema_decay=0.99):
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
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"Invalid ema_decay value: {ema_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, update_proj_gap=update_proj_gap, alpha=alpha,
                        use_cuda=use_cuda, ema_decay=ema_decay)
        super(GaLore, self).__init__(params, defaults)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __setstate__(self, state):
        super(GaLore, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
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
                    state['Q'] = None

                self._update_step(p, grad, group, state)

        return loss

    def _update_step(self, p: torch.Tensor, grad: torch.Tensor, group: dict, state: dict):
        """
        Performs a single update step for a parameter.

        Args:
            p (torch.Tensor): The parameter to update
            grad (torch.Tensor): The gradient of the parameter
            group (dict): The parameter group
            state (dict): The state of the parameter
        """
        # Update step count
        state['step'] += 1

        # Get hyperparameters
        beta1, beta2 = group['betas']
        rank = group['rank']
        update_proj_gap = group['update_proj_gap']

        # Update projection matrices if needed
        if state['step'] % update_proj_gap == 0 or state['P'] is None:
            P, Q = self._compute_projection_matrices(grad, rank)
            if state['P'] is None:
                state['P'], state['Q'] = P, Q
            else:
                # Apply EMA to projection matrices
                state['P'] = self._ema_update(state['P'], P, group['ema_decay'])
                state['Q'] = self._ema_update(state['Q'], Q, group['ema_decay'])

        # Project gradient
        proj_grad = self._project_gradient(grad, state['P'], state['Q'])

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
        update = self._project_back(update, state['P'], state['Q'])

        # Apply update
        p.add_(update, alpha=-step_size * group['alpha'])

    def _compute_projection_matrices(self, grad: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute projection matrices P and Q using SVD.

        Args:
            grad (torch.Tensor): The gradient tensor
            rank (int): The desired rank for projection

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: P and Q projection matrices
        """
        if self.use_cuda:
            grad = grad.cuda()

        try:
            U, S, V = torch.svd_lowrank(grad, q=rank)
        except RuntimeError:
            warnings.warn("SVD did not converge. Falling back to full SVD.", RuntimeWarning)
            U, S, V = torch.svd(grad, some=True)
            U = U[:, :rank]
            V = V[:, :rank]

        return U.to(self.device), V.to(self.device)

    def _project_gradient(self, grad: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Project gradient into low-rank space.

        Args:
            grad (torch.Tensor): The gradient tensor
            P (torch.Tensor): Left projection matrix
            Q (torch.Tensor): Right projection matrix

        Returns:
            torch.Tensor: Projected gradient
        """
        return torch.chain_matmul(P.t(), grad, Q)

    def _project_back(self, update: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Project update back to original space.

        Args:
            update (torch.Tensor): The update in low-rank space
            P (torch.Tensor): Left projection matrix
            Q (torch.Tensor): Right projection matrix

        Returns:
            torch.Tensor: Update in original space
        """
        return torch.chain_matmul(P, update, Q.t())

    @staticmethod
    def _ema_update(old: torch.Tensor, new: torch.Tensor, decay: float) -> torch.Tensor:
        """
        Perform an exponential moving average update.

        Args:
            old (torch.Tensor): The old tensor
            new (torch.Tensor): The new tensor
            decay (float): The decay factor

        Returns:
            torch.Tensor: The updated tensor
        """
        return old * decay + new * (1 - decay)

    def get_projection_matrices(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the current projection matrices for all parameters.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (P, Q) pairs for each parameter
        """
        return [(state['P'], state['Q']) for group in self.param_groups for p in group['params'] if p.grad is not None and 'P' in self.state[p]]
