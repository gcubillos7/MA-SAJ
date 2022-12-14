import torch as th
from typing import Dict, Iterable, Optional, Tuple, Union
from itertools import zip_longest

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T-1*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes (is dropped at the end)
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1)) # last return is target_qs if not finished
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        n_td_lambda = (1 - td_lambda) * gamma * target_qs[:, t-1] * (1 - terminated[:, t])
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (rewards[:, t] + n_td_lambda)
    # Returns lambda-return from t=0 to t=T-1, i.e. in [..., B*T-1, A]
    return ret [:, 0:-1]

# From: https://github.com/DLR-RM/stable-baselines3/blob/ef10189d80dbb2efb3b5391cba4eb3c2ab5c7aae/stable_baselines3/common/utils.py#L413
def polyak_update(
    params: Iterable[th.nn.Parameter],
    target_params: Iterable[th.nn.Parameter],
    tau: float,
) -> None:
    """
    
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)

    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)

# From: https://github.com/DLR-RM/stable-baselines3/blob/ef10189d80dbb2efb3b5391cba4eb3c2ab5c7aae/stable_baselines3/common/utils.py#L413
def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo