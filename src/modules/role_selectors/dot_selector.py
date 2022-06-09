import torch.nn as nn
import torch.nn.functional as F

import torch as th
from torch.distributions import Categorical


class DotSelector(nn.Module):
    def __init__(self, input_shape, args):
        super(DotSelector, self).__init__()
        self.args = args

        self.critic_encoder = nn.Sequential( nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim),
                                     nn.ReLU(),           
                                     nn.Linear(2 * args.rnn_hidden_dim, args.action_latent_dim))

                                     
        self.v_net  = nn.Sequential( nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim),
                                     nn.Linear(2 * args.rnn_hidden_dim, args.action_latent_dim)) 

        self.alpha = 1.0

    def critic_q(self, inputs, role_latent):

        x = self.critic_encoder(inputs) # [bs, action_latent_dim]
        x = x.unsqueeze(-1)
        role_latent_reshaped = role_latent.unsqueeze(0).repeat(x.shape[0], 1, 1) #

        role_q = th.bmm(role_latent_reshaped, x).squeeze()


    def forward(self, inputs, role_latent):
        

        return role_q

    def select_role(self, role_advantages, alpha, test_mode=False, t_env=None):

        logp = F.log_softmax(role_advantages/alpha)

        return logp


        # if test_mode:
        #     # Greedy action selection only
        #     self.epsilon = 0.0

        # # mask actions that are excluded from selection
        # masked_q_values = role_qs.detach().clone()

        # # choose random role
        # random_numbers = th.rand_like(role_qs[:, 0])
        # pick_random = (random_numbers < self.epsilon).long()
        # random_roles = Categorical(th.ones(role_qs.shape).float().to(self.args.device)).sample().long()

        # # choose randomly if pick radom else choose greedy
        # picked_roles = pick_random * random_roles + (1 - pick_random) * masked_q_values.max(dim=1)[1]
        # # [bs, 1]

        # return index of chosen role for each agent
        return picked_roles

    def update_alpha(self):
        raise NotImplementedError