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




