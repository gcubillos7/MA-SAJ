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

    def train_ma_saj(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        roles_shape_o = batch["roles"][:, :-1].shape
        role_at = int(np.ceil(roles_shape_o[1] / self.role_interval))
        role_t = role_at * self.role_interval

    def get_policy(self, batch):
        mac = self.mac
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        return mac_out

    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        avail_actions = batch["avail_actions"]

        alpha: float = max(0.05, 0.5 - t_env / 200000)  # linear decay

        mac_out, _ = self.get_policy(batch)

        # Mask out unavailable actions, normalise (as in action selection)
        mac_out[avail_actions == 0] = 1e-10
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        # TODO: Replace by policy
        pi = mac_out[:, :-1].clone()
        pi = pi.reshape(-1, self.n_actions)
        log_pi = th.log(pi)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)
        q_vals = th.min(q_vals1, q_vals2)

        pi = mac_out[:, :-1].reshape(-1, self.n_actions)
        entropies = - (pi * log_pi).sum(dim=-1)

        # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
        pol_target = (pi * (alpha * log_pi - q_vals[:, :-1].reshape(-1, self.n_actions))).sum(dim=-1)

        policy_loss = (pol_target * mask).sum() / mask.sum()

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.train_actor(batch, t_env, episode_num)
        self.train_critic(batch, t_env)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def train_critic(self, batch, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        states = batch["state"]

        mixer1 = self.mixer1
        mixer2 = self.mixer2

        alpha: float = max(0.05, 0.5 - t_env / 200000)  # linear decay

        # Actor
        mac_out, _ = self.get_policy(batch)  # Concat over time

        # Mask out unavailable actions
        mac_out[avail_actions == 0] = 0.0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        t_mac_out = mac_out.clone().detach()
        pi = t_mac_out

        # sample actions for next timesteps
        next_actions = Categorical(pi).sample().long().unsqueeze(3)

        next_actions_onehot = th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))
        if self.args.use_cuda:
            next_actions_onehot = next_actions_onehot.cuda()
        next_actions_onehot = next_actions_onehot.scatter_(3, next_actions, 1)

        pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:, 1:]
        pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        target_q_vals1 = self.target_critic1.forward(target_inputs).detach()
        target_q_vals2 = self.target_critic2.forward(target_inputs).detach()

        # directly caculate the values by definition
        next_vs1 = th.logsumexp(target_q_vals1 / alpha, dim=-1) * alpha
        next_vs2 = th.logsumexp(target_q_vals2 / alpha, dim=-1) * alpha

        next_chosen_qvals1 = th.gather(target_q_vals1, dim=3, index=next_actions).squeeze(3)
        next_chosen_qvals2 = th.gather(target_q_vals2, dim=3, index=next_actions).squeeze(3)

        target_qvals1 = self.target_mixer1(next_chosen_qvals1, states, actions=next_actions_onehot, vs=next_vs1)
        target_qvals2 = self.target_mixer2(next_chosen_qvals2, states, actions=next_actions_onehot, vs=next_vs2)

        target_qvals = th.min(target_qvals1, target_qvals2)

        # Calculate td-lambda targets
        target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma,
                                           self.args.td_lambda)
        targets = target_v - alpha * log_pi_taken.mean(dim=-1, keepdim=True)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)

        # directly caculate the values by definition
        vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha
        vs2 = th.logsumexp(q_vals2 / alpha, dim=-1) * alpha

        q_taken1 = th.gather(q_vals1[:, :-1], dim=3, index=actions).squeeze(3)
        q_taken2 = th.gather(q_vals2[:, :-1], dim=3, index=actions).squeeze(3)

        q_taken1 = mixer1(q_taken1, states[:, :-1], actions=actions_onehot, vs=vs1[:, :-1])
        q_taken2 = mixer2(q_taken2, states[:, :-1], actions=actions_onehot, vs=vs2[:, :-1])

        td_error1 = q_taken1 - targets.detach()
        td_error2 = q_taken2 - targets.detach()

        mask = mask.expand_as(td_error1)

        # 0-out the targets that came from padded data
        masked_td_error1 = td_error1 * mask
        loss1 = (masked_td_error1 ** 2).sum() / mask.sum()
        masked_td_error2 = td_error2 * mask
        loss2 = (masked_td_error2 ** 2).sum() / mask.sum()

        # Optimise
        self.c_optimiser1.zero_grad()
        loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
        self.c_optimiser1.step()

        self.c_optimiser2.zero_grad()
        loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
        self.c_optimiser2.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss1.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

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
        inputs = []
        inputs.append(batch["obs"][:])
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def train_role(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        # role_avail_actions = batch["role_avail_actions"]
        roles_shape_o = batch["roles"][:, :-1].shape
        role_at = int(np.ceil(roles_shape_o[1] / self.role_interval))
        role_t = role_at * self.role_interval

        roles_shape = list(roles_shape_o)
        roles_shape[1] = role_t
        roles = th.zeros(roles_shape).to(self.device)
        roles[:, :roles_shape_o[1]] = batch["roles"][:, :-1]
        roles = roles.view(batch.batch_size, role_at, self.role_interval, self.n_agents, -1)[:, :, 0]

        # Calculate estimated Q-Values
        mac_out = []
        role_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, role_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                role_out.append(role_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        role_out = th.stack(role_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_role_qvals = th.gather(role_out, dim=3, index=roles.long()).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_role_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_role_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                target_role_out.append(target_role_outs)

        target_role_out.append(th.zeros(batch.batch_size, self.n_agents, self.mac.n_roles).to(self.device))
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_role_out = th.stack(target_role_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # target_mac_out[role_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            # mac_out_detach[role_avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            role_out_detach = role_out.clone().detach()
            role_out_detach = th.cat([role_out_detach[:, 1:], role_out_detach[:, 0:1]], dim=1)
            cur_max_roles = role_out_detach.max(dim=3, keepdim=True)[1]
            target_role_max_qvals = th.gather(target_role_out, 3, cur_max_roles).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_role_max_qvals = target_role_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.role_mixer is not None:
            state_shape_o = batch["state"][:, :-1].shape
            state_shape = list(state_shape_o)
            state_shape[1] = role_t
            role_states = th.zeros(state_shape).to(self.device)
            role_states[:, :state_shape_o[1]] = batch["state"][:, :-1].detach().clone()
            role_states = role_states.view(batch.batch_size, role_at,
                                           self.role_interval, -1)[:, :, 0]
            chosen_role_qvals = self.role_mixer(chosen_role_qvals, role_states)
            role_states = th.cat([role_states[:, 1:], role_states[:, 0:1]], dim=1)
            target_role_max_qvals = self.target_role_mixer(target_role_max_qvals, role_states)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        rewards_shape = list(rewards.shape)
        rewards_shape[1] = role_t
        role_rewards = th.zeros(rewards_shape).to(self.device)
        role_rewards[:, :rewards.shape[1]] = rewards.detach().clone()
        role_rewards = role_rewards.view(batch.batch_size, role_at,
                                         self.role_interval).sum(dim=-1, keepdim=True)
        # role_terminated
        terminated_shape_o = terminated.shape
        terminated_shape = list(terminated_shape_o)
        terminated_shape[1] = role_t
        role_terminated = th.zeros(terminated_shape).to(self.device)
        role_terminated[:, :terminated_shape_o[1]] = terminated.detach().clone()
        role_terminated = role_terminated.view(batch.batch_size, role_at, self.role_interval).sum(dim=-1, keepdim=True)
        # role_terminated
        role_targets = role_rewards + self.args.gamma * (1 - role_terminated) * target_role_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        role_td_error = (chosen_role_qvals - role_targets.detach())

        mask = mask.expand_as(td_error)
        mask_shape = list(mask.shape)
        mask_shape[1] = role_t
        role_mask = th.zeros(mask_shape).to(self.device)
        role_mask[:, :mask.shape[1]] = mask.detach().clone()
        role_mask = role_mask.view(batch.batch_size, role_at, self.role_interval, -1)[:, :, 0]

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        masked_role_td_error = role_td_error * role_mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        role_loss = (masked_role_td_error ** 2).sum() / role_mask.sum()
        loss += role_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        pred_obs_loss = None
        pred_r_loss = None
        pred_grad_norm = None
        if self.role_action_spaces_updated:
            # train action encoder
            no_pred = []
            r_pred = []
            for t in range(batch.max_seq_length):
                no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                no_pred.append(no_preds)
                r_pred.append(r_preds)
            no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
            r_pred = th.stack(r_pred, dim=1)[:, :-1]
            no = batch["obs"][:, 1:].detach().clone()
            repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)

            pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
            pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimiser.zero_grad()
            pred_loss.backward()
            pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
            self.action_encoder_optimiser.step()

            if t_env > self.args.role_action_spaces_update_start:
                self.mac.update_role_action_spaces()
                if 'noar' in self.args.mac:
                    self.target_mac.role_selector.update_roles(self.mac.n_roles)
                self.role_action_spaces_updated = False
                self._update_targets()
                self.last_target_update_episode = episode_num

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", (loss - role_loss).item(), t_env)
            self.logger.log_stat("role_loss", role_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if pred_obs_loss is not None:
                self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
                self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
                self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("role_q_taken_mean",
                                 (chosen_role_qvals * role_mask).sum().item() / (
                                         role_mask.sum().item() * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
