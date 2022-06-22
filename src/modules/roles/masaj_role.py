import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MASAJRole(nn.Module):
    def __init__(self, args):
        super(MASAJRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.act_limit = args.act_limit

        self.mu_layer = nn.Linear(args.rnn_hidden_dim, args.action_latent_dim)
        self.log_std_layer = nn.Linear(args.rnn_hidden_dim, args.action_latent_dim)

        self.prior = None
        self.use_latent_normal = args.use_latent_normal

    def forward(self, hidden):
        latent_mu = self.mu_layer(hidden)
        latent_log_std = self.log_std_layer(hidden)
        latent_log_std = th.clamp(latent_log_std, LOG_STD_MIN, LOG_STD_MAX)
        latent_std = th.exp(latent_log_std)

        return latent_mu, latent_std

    def update_prior(self, prior):
        self.prior = prior

class MASAJRoleDiscrete(nn.Module):
    def __init__(self, args):
        super(MASAJRoleDiscrete, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.fc = nn.Linear(args.rnn_hidden_dim, args.action_latent_dim)
        self.action_space = th.ones(args.n_actions).to(args.device)

    def forward(self, h, action_latent):
        role_key = self.fc(h)  # [bs, action_latent] [n_actions, action_latent]
        role_key = role_key.unsqueeze(-1)
        action_latent_reshaped = action_latent.unsqueeze(0).repeat(role_key.shape[0], 1, 1)

        dot = th.bmm(action_latent_reshaped, role_key).squeeze()

        return dot

    def update_action_space(self, new_action_space):
        self.action_space = th.Tensor(new_action_space).to(self.args.device).float()

# class SquashedGaussianMLPActor(nn.Module):
#     """
#     From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
#     """
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
#         self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.act_limit = act_limit
#
#     def forward(self, obs, deterministic=False, with_logprob=True):
#         net_out = self.net(obs)
#         mu = self.mu_layer(net_out)
#         log_std = self.log_std_layer(net_out)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)
#
#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             # Only used for evaluating policy at test time.
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()
#
#         if with_logprob:
#             # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
#             # NOTE: The correction formula is a little bit magic. To get an understanding
#             # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
#             # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
#             # Try deriving it yourself as a (very difficult) exercise. :)
#
#             logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
#
#         else:
#             logp_pi = None
#
#         pi_action = torch.tanh(pi_action)
#         pi_action = self.act_limit * pi_action
#
#         return pi_action, logp_pi
