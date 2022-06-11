import torch.nn as nn
import torch.nn.functional as F

import torch as th
#from torch.distributions import Categorical


class DotSelector(nn.Module):
    def __init__(self, input_shape, args):
        super(DotSelector, self).__init__()
        self.args = args

        self.critic_encoder = nn.Sequential(nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(2 * args.rnn_hidden_dim, args.action_latent_dim))

        self.alpha = 1.0

    def critic_q(self, inputs, role_latent):
        x = self.critic_encoder(inputs)  # [bs, action_latent_dim]
        x = x.unsqueeze(-1)
        role_latent_reshaped = role_latent.unsqueeze(0).repeat(x.shape[0], 1, 1)  #

        role_q = th.bmm(role_latent_reshaped, x).squeeze()
        return role_q

    def forward(self, inputs, role_latent):
        role_q = self.critic_q(inputs, role_latent)
        role_v = th.logsumexp(role_q / self.alpha, dim=-1) * self.alpha
        return role_q, role_v

    def select_role(self, role_outputs, alpha, test_mode=False, t_env=None):
        role_q, role_v = role_outputs
        role_advantages = role_q - role_v
        log_p = F.log_softmax(role_advantages / alpha)

        return log_p

    def update_alpha(self, alpha):
        self.alpha = alpha
