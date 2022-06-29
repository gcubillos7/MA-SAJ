import copy

import numpy as np
import torch as th
# import torch.nn.functional as F
# from src.modules.critics.fop import FOPCritic
# from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.masaj import MASAJCritic, MASAJRoleCritic
from modules.mixers.fop import FOPMixer
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from modules.critics.value import ValueNet, RoleValueNet


# rnn critic https://github.com/AnujMahajanOxf/MAVEN/blob/master/maven_code/src/modules/critics/coma.py
# Role Selector -> Q para cada rol, para obs cada k steps (product dot entre centroid de clusters y output de rnn)
# Mixing Net para Rol Selector (FOP) -> (lambda net para los roles, mix net) -> Q para cada rol -> value con definition
# using Q discrete
# Se usa 
class MASAJ_Learner:
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

        self.critic1 = MASAJCritic(scheme, args)
        self.critic2 = MASAJCritic(scheme, args)

        self.mixer1 = FOPMixer(args)
        self.mixer2 = FOPMixer(args)

        self.target_mixer1 = copy.deepcopy(self.mixer1)
        self.target_mixer2 = copy.deepcopy(self.mixer2)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters()) + list(
            self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())

        self.value = ValueNet(scheme, args)
        # self.target_value = copy.deepcopy(self.value)
        if args.use_role_value:
            self.role_value = RoleValueNet(scheme, args)
            # self.target_role_value = copy.deepcopy(self.role_value)
            self.value_params = list(self.value.parameters()) + list(self.role_value.parameters())
        else:
            self.value_params = list(self.value.parameters())

        self.agent_params = list(mac.parameters())

        # Use FOP mixer
        self.role_mixer1 = FOPMixer(args)
        self.role_mixer2 = FOPMixer(args)

        self.role_target_mixer1 = copy.deepcopy(self.role_mixer1)
        self.role_target_mixer2 = copy.deepcopy(self.role_mixer2)

        # Use rnn + dot product --> Q, V using definition
        self.role_critic1 = MASAJRoleCritic(scheme, args)
        self.role_critic2 = MASAJRoleCritic(scheme, args)

        self.role_target_critic1 = copy.deepcopy(self.role_critic1)
        self.role_target_critic2 = copy.deepcopy(self.role_critic2)

        self.role_critic_params1 = list(self.role_critic1.parameters()) + list(self.role_mixer1.parameters())
        self.role_critic_params2 = list(self.role_critic2.parameters()) + list(self.role_mixer2.parameters())

        self.p_optimizer = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.c_optimizer1 = RMSprop(params=self.critic_params1 + self.role_critic_params1, lr=args.c_lr,
                                    alpha=args.optim_alpha,
                                    eps=args.optim_eps)

        self.c_optimizer2 = RMSprop(params=self.critic_params2 + self.role_critic_params2, lr=args.c_lr,
                                    alpha=args.optim_alpha,
                                    eps=args.optim_eps)

        self.val_optimizer = RMSprop(params=self.value_params, lr=args.v_lr, alpha=args.optim_alpha,
                                     eps=args.optim_eps)

        self.role_interval = args.role_interval
        self.device = args.device

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # mask = batch["filled"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # if mask.sum() == 0:
        #     self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
        #     self.logger.console_logger.error("Learner: mask.sum() == 0 at t_env {}".format(t_env))
        #     return

        self.train_encoder(batch, t_env)
        self.train_actor(batch, t_env)
        self.train_critic(batch, t_env)
        self.train_decoder(batch, t_env)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _get_policy(self, batch, mac):
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
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:  # role out skips last element
                # (avoid evaluating roles with 1 step)
                role_out.append(role_outs)

        mac_out = th.stack(mac_out, dim=1)
        role_out = th.stack(role_out, dim=1)

        # Return output of policy for each agent/role
        return mac_out, role_out

    def _get_joint_q_target(self, target_inputs, target_inputs_role, states, role_states, next_action, next_role,
                            alpha):
        """
        Get Q Joint Target
        # Input Shape shape [Bs, T,...] [Bs, TRole,...] (for TD lambda) 
        # Output Shape [Bs, T,...] [Bs, TRole,...] (for TD lambda) 
        """
        with th.no_grad():
            q_vals1 = self.target_critic1.forward(target_inputs, next_action)
            q_vals2 = self.target_critic2.forward(target_inputs, next_action)

            vs1 = self.value(target_inputs)
            vs2 = self.value(target_inputs)

            # Get Q and V values for roles
            if self.args.use_role_value:
                q_vals1_role = self.role_target_critic1.forward(target_inputs_role, next_role).detach()
                q_vals2_role = self.role_target_critic2.forward(target_inputs_role, next_role).detach()
                v_role1 = self.role_value(target_inputs)
                v_role2 = self.role_value(target_inputs)
            else:
                q_vals1_role = self.role_target_critic1.forward(target_inputs_role).detach()
                q_vals2_role = self.role_target_critic2.forward(target_inputs_role).detach()
                v_role1 = th.logsumexp(q_vals1_role / alpha, dim=-1) * alpha
                v_role2 = th.logsumexp(q_vals2_role / alpha, dim=-1) * alpha

            # Get Q joint for actions (using individual Qs and Vs)
            q_vals1 = self.target_mixer1(q_vals1, states, actions=next_action, vs=vs1)
            q_vals2 = self.target_mixer2(q_vals2, states, actions=next_action, vs=vs2)
            target_q_vals = th.min(q_vals1, q_vals2)

            # Get Q joint for roles (using individual Qs and Vs)
            q_vals1_role = self.target_mixer1(q_vals1_role, role_states, actions=next_role, vs=v_role1)
            q_vals2_role = self.target_mixer2(q_vals2_role, role_states, actions=next_role, vs=v_role2)
            target_q_vals_role = th.min(q_vals1_role, q_vals2_role)

        return target_q_vals, target_q_vals_role

    def _to_role_tensor(self, tensor, role_t, T_max_1):
        """
        Create a tensor representing roles each time step, the output is padded to be of size role_t
        """
        tensor_shape = tensor.shape
        roles_shape = list(tensor_shape)
        roles_shape[1] = role_t
        tensor_out = th.zeros(roles_shape).to(self.device)
        tensor_out[:, :T_max_1] = tensor.detach().clone()

        return tensor_out

    def _build_role_rollout(self, rewards, states, roles_taken, terminated, mask):
        """
        # role_out already missing last?
        # Use batch to build role inputs
        Input: Rewards [B, T-1], states [B, T-1], roles [B, T-1], terminated [B, T-1]
        Output: Roles [B, RoleT, role_interval], Roles States [B, RoleT, role_interval, -1], Roles Terminated [B, RoleT,
         role_interval]
        """

        roles_shape_o = roles_taken.shape  # bs, T-1, agents
        bs = roles_shape_o[0]  # batch size
        T_max_1 = roles_shape_o[1]  # T - 1

        # Get role transitions from batch
        role_at = int(np.ceil(T_max_1 / self.role_interval))  # always the same size as role_out
        role_t = role_at * self.role_interval

        # roles (actions)
        roles = self._to_role_tensor(roles_taken, role_t, T_max_1)
        roles = roles.view(bs, role_at, self.role_interval, self.n_agents, -1)[:, :, 0]

        # role_states
        role_states = self._to_role_tensor(states, role_t, T_max_1)
        role_states = role_states.view(bs, role_at, self.role_interval, -1)[:, :, 0]

        # role_terminated
        role_terminated = self._to_role_tensor(terminated, role_t, T_max_1)
        role_terminated = role_terminated.view(bs, role_at, self.role_interval).sum(dim=-1, keepdim=True)

        # role_rewards
        role_rewards = self._to_role_tensor(rewards, role_t, T_max_1)
        role_rewards = role_rewards.view(bs, role_at, self.role_interval).sum(dim=-1, keepdim=True)

        # role_mask
        role_mask = self._to_role_tensor(mask, role_t, T_max_1)
        role_mask = role_mask.view(bs, role_at, self.role_interval, -1)[:, :, 0]

        return role_rewards, role_states, roles, role_terminated, role_mask

    def _get_joint_q(self, inputs, inputs_role, states, role_states, action, role, alpha):
        """
        Get joint q
        # Input shape shape [Bs, T,...] [Bs, TRole,...]
        # Output shape [Bs*T] [Bs*TRole, (None or N_roles)]
        """
        # Get Q and V values for actions
        q_vals1 = self.critic1.forward(inputs, action)  # last q value isn't used
        q_vals2 = self.critic2.forward(inputs, action)

        with th.no_grad():
            vs1 = self.value(inputs)
            vs2 = self.value(inputs)

        # Get Q and V values for roles
        if self.args.use_role_value:
            # Possible Bug
            q_vals1_role = self.role_critic1.forward(inputs_role, role)
            q_vals2_role = self.role_critic2.forward(inputs_role, role)
            q_role_taken1 = q_vals1_role
            q_role_taken2 = q_vals2_role
            with th.no_grad():
                v_role1 = self.role_value(inputs)
                v_role2 = self.role_value(inputs)
        else:
            q_vals1_role = self.role_critic1.forward(inputs_role)  # [..., n_roles]
            q_vals2_role = self.role_critic2.forward(inputs_role)  # [..., n_roles]
            q_role_taken1 = th.gather(q_vals1_role, dim=3, index=action).squeeze(3)
            q_role_taken2 = th.gather(q_vals2_role, dim=3, index=action).squeeze(3)
            with th.no_grad():
                v_role1 = th.logsumexp(q_vals1_role / alpha, dim=-1) * alpha
                v_role2 = th.logsumexp(q_vals2_role / alpha, dim=-1) * alpha

        # Get Q joint for actions (using individual Qs and Vs)
        q_vals1 = self.mixer1(q_vals1, states, actions=action, vs=vs1)
        q_vals2 = self.mixer2(q_vals2, states, actions=action, vs=vs2)
        # q_vals = th.min(q_vals1, q_vals2)

        # Get Q joint for roles (using individual Qs and Vs)
        q_vals1_role = self.mixer1(q_role_taken1, role_states, actions=role, vs=v_role1)
        q_vals2_role = self.mixer2(q_role_taken2, role_states, actions=role, vs=v_role2)
        # q_vals_role = th.min(q_vals1_role, q_vals2_role)

        return (q_vals1, q_vals2), (q_vals1_role, q_vals2_role)

    def _get_q_values_no_grad(self, inputs, inputs_role, action, role):
        """
        Get flattened individual Q values
        """
        with th.no_grad():
            # Get Q values
            q_vals1 = self.critic1.forward(inputs, action)
            q_vals2 = self.critic2.forward(inputs, action)
            q_vals = th.min(q_vals1, q_vals2)  # Q(T,a) -> Q [Batch_size, seq, n_agents]
            q_vals = q_vals.reshape(-1)
            q_vals1_role = self.role_critic1.forward(inputs_role, role)
            q_vals2_role = self.role_critic2.forward(inputs_role, role)
            q_vals_role = th.min(q_vals1_role, q_vals2_role)  # Q(T,a)

            if self.args.use_role_value:
                # inputs = self.role_critic1._inputs(batch, bs, max_t)
                q_vals_role = q_vals_role.reshape(-1)
            else:
                q_vals_role = q_vals_role.reshape(-1, self.mac.n_roles)

        return q_vals, q_vals_role

    def train_encoder(self, batch, t_env):

        raise NotImplementedError

    def train_decoder(self, batch, t_env):

        raise NotImplementedError

    def train_actor(self, batch, t_env):
        """
        Update actor and value nets as in SAC (Haarjona)
        https://github.com/haarnoja/sac/blob/master/sac/algos/sac.py  
        Add regularization term for implicit constraints 
        Mixer isn't used during policy improvement
        """
        # TODO: Add role implicit constraints in policy loss, (auxiliary KL)      
        # KL(P, Q) = H(P,Q) - H(P) 
        # Max Entropy Objective  J = ... + alpha * H(π(· |st)) 
        # At the start we follow constraints at the end we don't
        # Reverse KL Objective  J = ... - alpha *¨KL(π(· |st), π_role(· |st)) = ... + alpha * H(π(· |st))  - alpha *
        # H(π(· |st), π_role(· |st))
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        alpha: float = max(0.05, 0.5 - t_env / 200000)  # linear decay

        role_at = int(np.ceil((max_t - 1) / self.role_interval))  # always the same size as role_out
        role_t = role_at * self.role_interval
        role_mask = self._to_role_tensor(mask, role_t, max_t - 1)
        role_mask = role_mask.view(bs, role_at, self.role_interval, -1)[:, :, 0]

        # [ep_batch.batch_size, max_t, self.n_agents, -1]
        mac_out, role_out = self._get_policy(batch, self.mac)

        # select action
        # get log p of actions
        role, log_p_role = role_out

        # [batch.batch_size, max_t, self.n_agents]
        action, log_p_action = mac_out

        log_p_action = log_p_action[:, :-1].clone()  # remove last step 
        log_p_action = log_p_action.reshape(-1)
        action = action[:, :-1]

        # inputs are shared between v's and q's
        # Make sure that critic stacks actions from the inside
        inputs = self.critic1._build_inputs(batch, bs, max_t)
        # TODO: Role critic should be in charge of building its inputs
        inputs_role = self.role_critic1._build_inputs(batch, bs, max_t)

        # inputs_role = inputs[:, :-1] --> but output is every role_interval steps 
        # if self.role_critic1 is fc: 
        # inputs_role = inputs[:, :-1][: ,::self.role_interval] --> output is same size as input

        # inputs [BS, T-1, ...] --> Outputs: [BS*T-1] [BS*TRole, (None or N_roles)]
        q_vals, q_vals_role = self._get_q_values_no_grad(inputs[:, :-1], inputs_role, action, role)

        # Get values for act (is not necessary, but it helps with stability)

        v_actions = self.value(
            inputs[:, :-1])  # inputs [BS, T-1, ...] --> Outputs: [BS*T-1] [BS*TRole, (None or N_roles)]
        v_actions = v_actions.reshape(-1)

        entropies = - (th.exp(log_p_action) * log_p_action).sum(dim=-1)

        act_target = (alpha * log_p_action - q_vals).sum(dim=-1)
        act_loss = (act_target * mask).sum() / mask.sum()

        #  Eq 10 in FOP Paper
        v_act_target = ((v_actions - (q_vals - alpha * log_p_action)) ** 2).sum(dim=-1)

        v_act_loss = (v_act_target * mask).sum() / mask.sum()

        # As roles are discrete we don't really need a value net as we can estimate V directly
        if self.args.use_role_value:
            # Shape is flattened
            log_p_role = log_p_role[:, :-1]
            log_p_role = log_p_role.reshape(-1)
            role = role[:, :-1]
            # Move V towards Q
            v_role = self.role_value(inputs_role).reshape(-1)
            v_role = v_role.reshape(-1)
            role_target = (alpha * log_p_role - q_vals_role).sum(dim=-1)
            role_loss = (role_target * role_mask).sum() / role_mask.sum()

            v_role_target = ((v_role - (q_vals_role - alpha * log_p_role)) ** 2).sum(dim=-1)
            v_role_loss = (v_role_target * role_mask).sum() / role_mask.sum()
        else:
            # Get Q for each role
            log_p_role_copy = log_p_role[:, :-1].clone()
            log_p_role_copy = log_p_role_copy.reshape(-1, self.mac.n_roles)
            role = th.exp(log_p_role_copy)  # Get p instead of log_p
            log_p_role = log_p_role[:, :-1].reshape(-1, self.mac.n_roles)
            # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
            role_target = (role * (alpha * log_p_role - q_vals_role.reshape(-1, self.mac.n_roles))).sum(
                dim=-1)
            role_loss = (role_target * role_mask).sum() / role_mask.sum()
            # The val net of roles isn't updated
            v_role_loss = 0

        loss_policy = act_loss + role_loss
        loss_value = v_act_loss + v_role_loss
        if self.args.continuos_actions:
            kl_loss = self.mac.get_kl_loss()[:, :-1]
            masked_kl_loss = (kl_loss * mask).sum() / mask.sum()
            loss_policy += masked_kl_loss
        # Optimize values
        self.val_optimizer.zero_grad()
        loss_value.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
        self.val_optimizer.step()

        # Optimize policy
        self.p_optimizer.zero_grad()
        loss_policy.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_policy", loss_policy.item(), t_env)
            self.logger.log_stat("loss_value", loss_value.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train_critic(self, batch, t_env):
        alpha: float = max(0.05, 0.5 - t_env / 200000)
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        states = batch["state"]
        actions_taken = batch["actions"][:, :-1]
        roles_taken = batch["roles"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        role_rewards, role_states, roles, role_terminated, role_mask = self._build_role_rollout(rewards, states,
                                                                                                roles_taken, terminated,
                                                                                                mask)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        inputs_role = self.role_critic1._build_inputs(batch, bs, max_t)

        # Sample roles according to current policy and get their log probabilities
        mac_out, role_out = self._get_policy(batch, self.mac)

        # select action
        # get log p of actions
        next_role, log_p_role = role_out

        # [batch.batch_size, max_t, self.n_agents]
        next_action, log_p_action = mac_out

        # Find Q values of actions and roles according to current policy
        target_act_joint_q, target_role_joint_q = self._get_joint_q_target(inputs, inputs_role, states, role_states,
                                                                           next_action, next_role, alpha)

        # TD lambda deals with moving targets 1 step forward
        target_v_act = build_td_lambda_targets(rewards, terminated, mask, target_act_joint_q, self.n_agents,
                                               self.args.gamma,
                                               self.args.td_lambda)

        target_v_role = build_td_lambda_targets(role_rewards, role_terminated, role_mask, target_role_joint_q,
                                                self.n_agents, self.args.gamma,
                                                self.args.td_lambda)

        #  Eq 9 in FOP Paper
        targets_act = target_v_act - alpha * log_p_action.mean(dim=-1, keepdim=True)
        targets_role = target_v_role - alpha * log_p_role.mean(dim=-1, keepdim=True)

        # Find Q values of actions and roles taken in batch
        q_act_taken, q_role_taken = self._get_joint_q(inputs[:, :-1], inputs_role[:, :-1], states[:, :-1],
                                                      role_states[:, :-1],
                                                      actions_taken[:, :-1], roles_taken[:, :-1], alpha)

        q1_act_taken, q2_act_taken = q_act_taken  # double q

        q1_role_taken, q2_role_taken = q_role_taken  # double q

        td_error1_role = q1_role_taken - targets_role.detach()
        td_error2_role = q2_role_taken - targets_role.detach()

        td_error1_act = q1_act_taken - targets_act.detach()
        td_error2_act = q2_act_taken - targets_act.detach()

        # 0-out the targets that came from padded data
        role_mask = role_mask.expand_as(td_error1_role)
        masked_td_error1_role = td_error1_role * role_mask
        loss1 = (masked_td_error1_role ** 2).sum() / role_mask.sum()
        masked_td_error2_role = td_error2_role * role_mask
        loss2 = (masked_td_error2_role ** 2).sum() / role_mask.sum()

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error1_act)
        masked_td_error1 = td_error1_act * mask
        loss1 += (masked_td_error1 ** 2).sum() / mask.sum()
        masked_td_error2 = td_error2_act * mask
        loss2 += (masked_td_error2 ** 2).sum() / mask.sum()

        # Optimize
        self.c_optimizer1.zero_grad()
        loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1 + self.role_critic_params1,
                                                self.args.grad_norm_clip)
        self.c_optimizer1.step()

        self.c_optimizer2.zero_grad()
        loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2 + self.role_critic_params2,
                                                self.args.grad_norm_clip)
        self.c_optimizer2.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss1.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (q1_act_taken * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets_act * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

            self.log_stats_t = t_env

        # RODE logging
        # if t_env - self.log_stats_t >= self.args.learner_log_interval:
        #     self.logger.log_stat("loss", (loss - role_loss).item(), t_env)
        #     self.logger.log_stat("role_loss", role_loss.item(), t_env)
        #     self.logger.log_stat("grad_norm", grad_norm, t_env)
        #     if pred_obs_loss is not None:
        #         self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
        #         self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
        #         self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm, t_env)
        #     mask_elems = mask.sum().item()
        #     self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
        #     self.logger.log_stat("q_taken_mean",
        #                          (chosen_action_q_vals * mask).sum().item() / (mask_elems * self.args.n_agents),
        #                          t_env)
        #     self.logger.log_stat("role_q_taken_mean",
        #                          (chosen_role_q_vals * role_mask).sum().item() / (role_mask.sum().item() *
        #                          self.args.n_agents), t_env)
        #     self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
        #                          t_env)
        #     self.log_stats_t = t_env

    def _update_targets(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_mixer1.load_state_dict(self.mixer1.state_dict())
        self.target_mixer2.load_state_dict(self.mixer2.state_dict())

        self.role_target_critic1.load_state_dict(self.role_critic1.state_dict())
        self.role_target_critic2.load_state_dict(self.role_critic2.state_dict())
        self.role_target_mixer1.load_state_dict(self.role_mixer1.state_dict())
        self.role_target_mixer2.load_state_dict(self.role_mixer2.state_dict())

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
        th.save(self.p_optimizer.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.c_optimizer1.state_dict(), "{}/critic_opt1.th".format(path))
        th.save(self.c_optimizer2.state_dict(), "{}/critic_opt2.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right, but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))

        self.p_optimizer.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimizer1.load_state_dict(
            th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimizer2.load_state_dict(
            th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
