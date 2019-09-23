import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist

def _flatten_tensors(tensors):
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def _unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

class DSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, model, update_period = 10, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, vrl=False, local=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DSGD, self).__init__(params, defaults)
        self.model = model
        self.vrl = vrl
        self.iter_cnt = 0
        if not local:
            update_period = 1
        self.update_period = update_period
        print("vrl:{}".format(self.vrl))


    def __setstate__(self, state):
        super(DSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def _update_params(self):

        with torch.no_grad():
            for group in self.param_groups:
                momentum = group['momentum']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if self.vrl:
                        param_state["last_param_buff"] = p.clone().detach_()
                    if momentum != 0:
                        #sync momentum
                        if 'momentum_buffer' not in param_state:
                            continue
                        else:
                            param_state['momentum_buffer'] = param_state['momentum_buffer']/dist.get_world_size()
                            # out_msg_list = _flatten_tensors(buf).clone().detach()
                            dist.all_reduce(param_state['momentum_buffer'], async_op=True)
            # sync model params
            self.model._sync_period()
            dist.barrier()
            

            # update delta grad
            for group in self.param_groups:
                lr = group["lr"]
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if self.vrl:
                        param_state["vrl_buff"] = param_state["vrl_buff"]+ 1.0/(lr*self.update_period)*(p - param_state["last_param_buff"])


    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # return 0
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if self.vrl:
                    if 'vrl_buff' not in param_state:
                        param_state['vrl_buff'] = torch.zeros_like(d_p).detach()
                    d_p = d_p - param_state['vrl_buff']
                p.data.add_(-group['lr'], d_p)
        self.iter_cnt += 1
        if self.iter_cnt % self.update_period ==0:
            self._update_params()
        return loss
