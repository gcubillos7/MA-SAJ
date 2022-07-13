import torch.nn as nn
import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialRoleSelector(nn.Module):
    def __init__(self, input_shape, args):
        super(MultinomialRoleSelector, self).__init__()
        self.args = args

        self.critic_encoder = nn.Sequential(nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(2 * args.rnn_hidden_dim, args.action_latent_dim))

        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                              args.epsilon_finish,
                                              args.epsilon_anneal_time,
                                              args.epsilon_anneal_time_exp,
                                              args.role_action_spaces_update_start,
                                              decay="linear")

        self.role_action_spaces_update_start = self.args.role_action_spaces_update_start

        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)

    def dot_product(self, inputs, role_latent):
        x = self.critic_encoder(inputs)  # [bs, action_latent_dim]
        x = x.unsqueeze(-1)
        role_latent_reshaped = role_latent.unsqueeze(0).repeat(x.shape[0], 1, 1)  #

        dot = th.bmm(role_latent_reshaped, x).squeeze()

        return dot

    def forward(self, inputs, role_latent):
        dot = self.dot_product(inputs, role_latent)

        return dot  # logits # TODO: logp = log softmax(logits)

    def select_role(self, role_pis, t_env, test_mode=False):
        # role_pis [bs*n_agents, n_roles] 
        if t_env is not None:
            self.epsilon = self.schedule.eval(t_env)
        dist = Categorical(role_pis)
        if test_mode and self.test_greedy:
            picked_roles = role_pis.max(dim=1)[1]
        else:
            picked_roles = dist.sample().long()
        log_p_role = dist.log_prob(picked_roles)
        return picked_roles, log_p_role


REGISTRY["multinomial_role"] = MultinomialRoleSelector

# def select_role(self, policies, test_mode=False, t_env=None):
#     self.epsilon = self.epsilon_schedule(t_env)
#
#     if test_mode:
#         # Greedy action selection only
#         self.epsilon = 0.0
#
#     # mask actions that are excluded from selection
#     masked_policies = policies.detach().clone()
#
#     random_numbers = th.rand_like(role_qs[:, 0])
#     pick_random = (random_numbers < self.epsilon).long()
#     random_roles = Categorical(th.ones(role_qs.shape).float().to(self.args.device)).sample().long()
#
#     picked_roles = pick_random * random_roles + (1 - pick_random) * masked_q_values.max(dim=1)[1]
#     # [bs, 1]
#     return picked_roles
#
# import torch.nn as nn
# import torch.nn.functional as F
#
# import torch as th
# from torch.distributions import Categorical
#
#
class DotSelector(nn.Module):
    def __init__(self, input_shape, args):
        super(DotSelector, self).__init__()
        self.args = args
        self.epsilon_start = self.args.epsilon_start
        self.epsilon_finish = self.args.role_epsilon_finish
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.epsilon_anneal_time_exp = self.args.epsilon_anneal_time_exp
        self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
        self.role_action_spaces_update_start = self.args.role_action_spaces_update_start
        self.epsilon_start_t = 0
        self.epsilon_reset = True

        self.fc1 = nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim)
        self.fc2 = nn.Linear(2 * args.rnn_hidden_dim, args.action_latent_dim)

        self.epsilon = 0.05

    def forward(self, inputs, role_latent):
        x = self.fc2(F.relu(self.fc1(inputs)))  # [bs, action_dim] [n_roles, action_dim] (bs may be bs*n_agents)
        x = x.unsqueeze(-1)
        role_latent_reshaped = role_latent.unsqueeze(0).repeat(x.shape[0], 1, 1)

        role_q = th.bmm(role_latent_reshaped, x).squeeze()
        return role_q

    def select_role(self, role_qs, test_mode=False, t_env=None):
        self.epsilon = self.epsilon_schedule(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = role_qs.detach().clone()

        random_numbers = th.rand_like(role_qs[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_roles = Categorical(th.ones(role_qs.shape).float().to(self.args.device)).sample().long()

        picked_roles = pick_random * random_roles + (1 - pick_random) * masked_q_values.max(dim=1)[1]
        # [bs, 1]
        return picked_roles

    def epsilon_schedule(self, t_env):
        if t_env is None:
            return 0.05

        if t_env > self.role_action_spaces_update_start and self.epsilon_reset:
            self.epsilon_reset = False
            self.epsilon_start_t = t_env
            self.epsilon_anneal_time = self.epsilon_anneal_time_exp
            self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time

        if t_env - self.epsilon_start_t > self.epsilon_anneal_time:
            epsilon = self.epsilon_finish
        else:
            epsilon = self.epsilon_start - (t_env - self.epsilon_start_t) * self.delta

        return epsilon
