import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.fop import FOPMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np
from torch.distributions import Categorical
from src.modules.critics.fop import FOPCritic
from src.utils.rl_utils import build_td_lambda_targets


# Role Selector -> Q para cada rol, para obs cada k pasos (producto punto entre centroide de clusters y salida de rnn)
# Mixing Net para Rol Selector (FOP) -> (lambda net para los roles, mix net) -> Q para cada rol -> value con definición usando Q discreto
# Se usa 
class FOP_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic1 = FOPCritic(scheme, args)
        self.critic2 = FOPCritic(scheme, args)

        self.mixer1 = FOPMixer(args)
        self.mixer2 = FOPMixer(args)

        self.target_mixer1 = copy.deepcopy(self.mixer1)
        self.target_mixer2 = copy.deepcopy(self.mixer2)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())

        self.agent_params = list(mac.parameters())

        self.p_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.c_optimiser1 = RMSprop(params=self.critic_params1, lr=args.c_lr, alpha=args.optim_alpha,
                                    eps=args.optim_eps)
        self.c_optimiser2 = RMSprop(params=self.critic_params2, lr=args.c_lr, alpha=args.optim_alpha,
                                    eps=args.optim_eps)

        # Use FOP mixer
        self.role_mixer1 = FOPMixer(args)
        self.role_mixer2 = FOPMixer(args)

        # Use rnn + dot product --> Q, V using definition
        self.role_critic1 = FOPCritic(scheme, args)
        self.role_critic2 = FOPCritic(scheme, args)

        self.target_role_critic1 = copy.deepcopy(self.role_critic1)
        self.target_role_critic2 = copy.deepcopy(self.role_critic2)

        self.role_critic_params1 = list(self.role_critic1.parameters()) + list(self.role_mixer1.parameters())
        self.role_critic_params2 = list(self.role_critic2.parameters()) + list(self.role_mixer2.parameters())

        self.r_c_optimiser1 = RMSprop(params=self.role_critic_params1, lr=args.c_lr, alpha=args.optim_alpha,
                                      eps=args.optim_eps)
        self.r_c_optimiser2 = RMSprop(params=self.role_critic_params2, lr=args.c_lr, alpha=args.optim_alpha,
                                      eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.train_encoder(batch, t_env)
        self.train_actor(batch, t_env)
        self.train_critic(batch, t_env)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def get_policy(self, batch, mac):
        """
        Returns
        mac_out: returns distribution of actions .log_p(actions)
        role_out: returns distribution over roles
        """
        # Get role policy and mac policy
        mac_out = []
        role_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, role_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                role_out.append(role_outs)

        mac_out = th.stack(mac_out, dim=1)
        role_out = th.stack(role_out, dim=1)

        # Return output of policy for each agent/role
        return mac_out, role_out

    def train_actor(self, batch, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        alpha: float = max(0.05, 0.5 - t_env / 200000)  # linear decay
        # [ep_batch.batch_size, max_t, self.n_agents, -1]
        mac_out, role_out = self.get_policy(batch, self.mac)
        # t_mac_out, t_role_out = self.get_policy(batch, self.target_mac)

        # select action
        # get log p of actions
        pi_role, log_pi_role = role_out
        # [batch.batch_size, max_t, self.n_agents]
        pi_a, log_pi_a = mac_out

        log_pi_a = log_pi_a[:, :-1].clone()
        log_pi_a = log_pi_a.reshape(-1)  # , self.n_actions)
        pi_a = pi_a[:, :-1].clone()
        pi_a = pi_a.reshape(-1)
        # log_pi = th.log(pi)
        log_pi_role = log_pi_role[:, :-1].clone()
        log_pi_role = log_pi_role.reshape(-1)  # , self.n_actions)
        pi_role = pi_role[:, :-1].clone()
        pi_role = pi_role.reshape(-1)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)
        q_vals = th.min(q_vals1, q_vals2) # q_vals_actions -> Q [Batch_size, seq, n_agents]

        inputs = self.role_critic1._build_inputs(batch, bs, max_t)
        q_vals1_role = self.role_critic1.forward(inputs)
        q_vals2_role = self.role_critic2.forward(inputs)
        q_vals_role = th.min(q_vals1_role, q_vals2_role)
        # q_vals_role -> Q [Batch_size, seq, n_agents] , Q [Batch_size, seq, n_roles, n_agents]
        # chosen_role_q_vals = th.gather(q_vals_role[:, :-1], dim=3, index=pi_role.long()).squeeze(3)
        # Remove the last dim

        entropies = - (th.exp(log_pi_a) * log_pi_a).sum(dim=-1)

        pol_target = (alpha * log_pi_a - q_vals[:, :-1]).sum(dim=-1)

        rol_target = (alpha * log_pi_role - q_vals_role[:, :-1]).sum(dim=-1)

        target = pol_target + rol_target
        loss = (target * mask).sum() / mask.sum()

        # Optimise
        self.p_optimiser.zero_grad()
        loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            # self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train_critic(self, batch, t_env):
        pass

    def _update_targets(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_mixer1.load_state_dict(self.mixer1.state_dict())
        self.target_mixer2.load_state_dict(self.mixer2.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic1.cuda()
        self.mixer1.cuda()
        self.target_critic1.cuda()
        self.target_mixer1.cuda()
        self.critic2.cuda()
        self.mixer2.cuda()
        self.target_critic2.cuda()
        self.target_mixer2.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.mixer2.state_dict(), "{}/mixer2.th".format(path))
        th.save(self.p_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.c_optimiser1.state_dict(), "{}/critic_opt1.th".format(path))
        th.save(self.c_optimiser2.state_dict(), "{}/critic_opt2.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))

        self.p_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimiser1.load_state_dict(
            th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimiser2.load_state_dict(
            th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))

    def build_inputs(self, batch, bs, max_t, actions_onehot):
        inputs = [batch["obs"][:]]
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs