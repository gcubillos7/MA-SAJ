import torch as th
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from .epsilon_schedules import DecayThenFlatSchedule
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

REGISTRY = {}

'''
class GumbelSoftmax():
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.eps = 1e-10
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False): 
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            U = th.rand(masked_policies.size()).cuda()
            y = masked_policies - th.log(-th.log(U + self.eps) + self.eps)
            y = F.softmax(y / 1, dim=-1)
            y[avail_actions == 0.0] = 0.0
            picked_actions = y.max(dim=2)[1]

        return picked_actions

REGISTRY["gumbel"] = GumbelSoftmax
'''


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                args.epsilon_finish, args.epsilon_anneal_time,
                                time_length_exp = args.epsilon_anneal_time_exp,
                                role_action_spaces_update_start = args.role_action_spaces_update_start,
                                decay="linear")

        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        if t_env is not None:
            self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              args.epsilon_anneal_time_exp,
                                              args.role_action_spaces_update_start,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        
        return picked_actions

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

        # self.act_limit = 1.0
        self.decoder = None
        self.prior = None
        self.use_latent_normal = args.use_latent_normal
        if not self.use_latent_normal:
            self.dkl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        else:
            self.dkl = kl_divergence

    def select_action(self, mu, sigma, t_env = None, test_mode=False):
        # expects the following input dimensionalities:
        # mu: [b x a x u]
        # sigma: [b x a x u]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(-1, self.args.n_agents, self.args.n_actions, self.args.n_actions)

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = th.distributions.MultivariateNormal(mu.view(-1,
                                                              mu.shape[-1]),
                                                      sigma.view(-1,
                                                                 mu.shape[-1],
                                                                 mu.shape[-1]))
            try:
                picked_actions = dst.sample().view(*mu.shape)
            except Exception as e:
                a = 5
                pass
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector


class GaussianLatentActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)
        self.prior = None
        self.decoder = None
        self.use_latent_normal = args.use_latent_normal
        if not self.use_latent_normal:
            self.dkl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        else:
            self.dkl = kl_divergence
        self.with_logprob = True
        self.threshold = nn.parameter.Parameter(th.tensor(0.3181), requires_grad=False)  # 2 times the variance same mu
        self.act_limit = args.act_limit

    def update_decoder(self, decoder):
        self.decoder = decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

    def select_action(self, mu, sigma, prior, test_mode=False):
        dkl_loss = None
        latent_dist = Normal(mu, sigma)
        latent_action = mu if test_mode else latent_dist.sample()
        log_p_latent = latent_dist.log_prob(latent_action).sum(dim=-1)
        if not test_mode and prior:
            if self.use_latent_normal:  # dkl distributions
                # [bs, action_latent] [n_actions, action_latent]
                dkl_loss = self.dkl(latent_dist, prior)
            else:
                sample = prior.sample()
                log_p_prior = prior.log_prob(sample).sum(dim=-1)
                dkl_loss = self.dkl(log_p_latent, log_p_prior)
            dkl_loss = th.max(dkl_loss, self.threshold)  # don't enforce the dkl inside the threshold

        if self.with_logprob:
            pi_action, log_p_pi = self.decoder(latent_action)
            log_p_pi += log_p_latent  # p_latent * p_action

            log_p_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            pi_action = self.decoder(latent_action)
            log_p_pi = None

        pi_action = th.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, log_p_pi, dkl_loss


REGISTRY["gaussian_latent"] = GaussianLatentActionSelector
