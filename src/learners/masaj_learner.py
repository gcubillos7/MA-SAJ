import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.masaj import MASAJCritic, MASAJRoleCritic
from modules.mixers.fop import FOPMixer
from utils.rl_utils import build_td_lambda_targets, polyak_update
from torch.optim import RMSprop, Adam
from modules.critics.value import ValueNet, RoleValueNet


# rnn critic https://github.com/AnujMahajanOxf/MAVEN/blob/master/maven_code/src/modules/critics/coma.py
# Role Selector -> Q para cada rol, para obs cada k steps (product dot entre centroid de clusters y output de rnn)
# Mixing Net para Rol Selector (FOP) -> (lambda net para los roles, mix net) -> Q para cada rol -> value con definition
# using Q discrete
# Se usa 
class MASAJ_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.continuous_actions = args.continuous_actions
        self.use_role_value = args.use_role_value
        self.logger = logger

        self.mac = mac
        self.mac.logger = logger

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_roles = args.n_roles
        self.n_role_clusters = args.n_role_clusters
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

        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())
        
        self.value_params = []
        self.role_action_spaces_updated = True
        if self.continuous_actions:
            self._get_policy_continuous
            self.train_encoder = self.train_encoder_continuous
            self.value = ValueNet(scheme, args)
            self.value_params += list(self.value.parameters())
            self.value = copy.deepcopy(self.value)
        else:
            self._get_policy = self._get_policy_discrete
            self.train_encoder = self.train_encoder_discrete

        if self.use_role_value:
            self.role_value = RoleValueNet(scheme, args)
            self.value_params += list(self.role_value.parameters())
            # self.target_role_value = copy.deepcopy(self.role_value)
        else:
            if self.n_roles != self.n_role_clusters:
                Warning('n_roles != n_role_clusters some clusters could be combined')

        self.agent_params = list(mac.parameters())

        # Use FOP mixer
        self.role_mixer1 = FOPMixer(args, n_actions=self.n_roles)
        self.role_mixer2 = FOPMixer(args, n_actions=self.n_roles)

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

        if self.use_role_value or self.continuous_actions:
            self.val_optimizer = RMSprop(params=self.value_params, lr=args.v_lr, alpha=args.optim_alpha,
                                         eps=args.optim_eps)

        self.role_interval = args.role_interval
        self.device = args.device

        self.action_encoder_params = list(self.mac.action_encoder_params())
        self.action_encoder_optimizer = RMSprop(params=self.action_encoder_params, lr=args.lr,
                                                alpha=args.optim_alpha, eps=args.optim_eps)
        self._build_ent_coeficient(args)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        self.train_encoder(batch, t_env, episode_num)
        alpha = self.get_alpha(t_env)
        self.train_critic(batch, t_env, alpha)
        self.train_actor(batch, t_env, alpha)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _get_policy_discrete(self,batch, mac, avail_actions, test_mode=False):
        """
        Returns
        mac_out: returns distribution of actions .log_p(actions)
        role_out: returns distribution over roles
        """
        # Get role policy and mac policy
        mac_out = []
        role_taken = []
        log_p_role = []
        role_avail_actions = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, role_outs = self.mac.forward(batch, t=t, test_mode=test_mode)
            role_avail_actions.append(self.mac.role_avail_actions)
            mac_out.append(agent_outs[0])
            if t % self.role_interval == 0 and (t < batch.max_seq_length - 1):  # role out skips last element
                role_taken.append(role_outs[0])
                log_p_role.append(role_outs[1])

        role_taken, log_p_role = (th.stack(role_taken, dim=1), th.stack(log_p_role, dim=1))
        
        mac_role_out = (role_taken, log_p_role)  # [...], [...]
        
        pi_act = th.stack(mac_out, dim=1)

        # normalize outputs 
        role_avail_actions = th.stack(role_avail_actions, dim=1)
        true_avail_actions = avail_actions * role_avail_actions
        # output is the full policy
        pi_act[true_avail_actions == 0] = 1e-11
        pi_act = pi_act / pi_act.sum(dim=-1, keepdim=True)
        pi_act[true_avail_actions == 0] = 1e-11
        
        pi = pi_act.clone()
        log_p_out = th.log(pi)
        mac_out = (pi_act, log_p_out)  # [..., n_actions], [..., n_actions]

        return mac_out, mac_role_out        

    def _get_policy_continuous(self,batch, mac, avail_actions, test_mode=False):
        """
        Returns
        mac_out: returns distribution of actions .log_p(actions)
        role_out: returns distribution over roles
        """
        # Get role policy and mac policy
        mac_out = []
        log_p_out = []
        role_taken = []
        log_p_role = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, role_outs = self.mac.forward(batch, t=t, test_mode=test_mode)
            mac_out.append(agent_outs[0])
            log_p_out.append(agent_outs[1])
            if t % self.role_interval == 0 and (t < batch.max_seq_length - 1):  # role out skips last element
                role_taken.append(role_outs[0])
                log_p_role.append(role_outs[1])

        role_taken, log_p_role = (th.stack(role_taken, dim=1), th.stack(log_p_role, dim=1))
        mac_role_out = (role_taken, log_p_role)  # [...], [...]

        # Outputs is action, log_p
        action_taken, log_p_action = (th.stack(mac_out, dim=1), th.stack(log_p_out, dim=1))
        mac_out = (action_taken, log_p_action)  # [...], [...]

        return mac_out, mac_role_out

    def _get_joint_q_target(self, target_inputs, target_inputs_role, states, role_states, next_action, next_role, next_role_role_onehot,
                            alpha):
        """
        Get Q Joint Target
        # Input Shape shape [Bs, T,...] [Bs, TRole,...] (for TD lambda) 
        # Output Shape [Bs, T,...] [Bs, TRole,...] (for TD lambda) 
        """

        with th.no_grad():
            if self.continuous_actions:
                next_action_input = next_action
                q_vals_taken1 = self.target_critic1.forward(target_inputs, next_action_input)  # [...]
                q_vals_taken2 = self.target_critic2.forward(target_inputs, next_action_input)  # [...]
                vs1 = self.value(target_inputs).detach()  # [...]
                vs2 = vs1  # [...]
            else:
                next_action_input = F.one_hot(next_action, num_classes=self.n_actions)
                q_vals1 = self.target_critic1.forward(target_inputs)  # [..., n_actions]
                q_vals2 = self.target_critic2.forward(target_inputs)  # [..., n_actions]

                q_vals_taken1 = th.gather(q_vals1, dim=3, index=next_action).squeeze(3)  # [...]
                q_vals_taken2 = th.gather(q_vals2, dim=3, index=next_action).squeeze(3)  # [...]

                vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha  # [...]
                vs2 = th.logsumexp(q_vals2 / alpha, dim=-1) * alpha  # [...]

            # Get Q joint for actions (using individual Qs and Vs)
            q_vals1 = self.target_mixer1(q_vals_taken1, states, actions=next_action_input, vs=vs1)  # collapses n_agents
            q_vals2 = self.target_mixer2(q_vals_taken2, states, actions=next_action_input, vs=vs2)  # collapses n_agents
            target_q_vals = th.min(q_vals1, q_vals2)

            # Get Q and V values for roles
            if self.args.use_role_value:
                q_role_taken1 = self.role_target_critic1.forward(target_inputs_role, next_role_role_onehot).detach()
                q_role_taken2 = self.role_target_critic2.forward(target_inputs_role, next_role_role_onehot).detach()
                v_role1 = self.role_value(target_inputs_role).detach()
                # v_role1 = self.target_role_value(target_inputs_role).detach()
                v_role2 = v_role1
            else:
                q_vals1_role = self.role_target_critic1.forward(target_inputs_role).detach()  # [..., n_roles]
                q_vals2_role = self.role_target_critic2.forward(target_inputs_role).detach()  # [..., n_roles]

                q_role_taken1 = th.gather(q_vals1_role, dim=3, index=next_role).squeeze(3)
                q_role_taken2 = th.gather(q_vals2_role, dim=3, index=next_role).squeeze(3)

                v_role1 = th.logsumexp(q_vals1_role / alpha, dim=-1) * alpha
                v_role2 = th.logsumexp(q_vals2_role / alpha, dim=-1) * alpha

            # Get Q joint for roles taken (using individual Qs and Vs)
            q_vals1_role = self.role_target_mixer1(q_role_taken1, role_states, actions = next_role_role_onehot,
                                                   vs=v_role1)  # collapses n_agents
            q_vals2_role = self.role_target_mixer2(q_role_taken2, role_states, actions = next_role_role_onehot,
                                                   vs=v_role2)  # collapses n_agents

            target_q_vals_role = th.min(q_vals1_role, q_vals2_role)

        return target_q_vals, target_q_vals_role

    def _get_joint_q(self, inputs, inputs_role, states, role_states, action, role, action_onehot, role_onehot, alpha):
        """
        Get joint q
        # Input shape shape [Bs, T,...] [Bs, TRole,...]
        # Output shape [Bs*T] [Bs*TRole, (None or N_roles)]
        """

        # Get Q and V values for actions
        if self.continuous_actions:
            action_input = action
            q_vals_taken1 = self.critic1.forward(inputs, action_input)  # last q value isn't used
            q_vals_taken2 = self.critic2.forward(inputs, action_input)  # [...]
            with th.no_grad():
                # vs1 = self.value(inputs).detach()
                vs1 = self.value(inputs).detach()
                vs2 = vs1
        else:
            action_input = action_onehot
            q_vals1 = self.critic1.forward(inputs)  # [..., n_actions]
            q_vals2 = self.critic2.forward(inputs)

            q_vals_taken1 = th.gather(q_vals1, dim=3, index=action).squeeze(3)
            q_vals_taken2 = th.gather(q_vals2, dim=3, index=action).squeeze(3)

            vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha
            vs2 = th.logsumexp(q_vals2 / alpha, dim=-1) * alpha

        # Get Q joint for actions (using individual Qs and Vs)
        q_vals1 = self.mixer1(q_vals_taken1, states, actions=action_input, vs=vs1)
        q_vals2 = self.mixer2(q_vals_taken2, states, actions=action_input, vs=vs2)

        # Get Q and V values for roles
        if self.use_role_value:
            # Possible Bug
            q_vals1_role = self.role_critic1.forward(inputs_role, role_onehot)
            q_vals2_role = self.role_critic2.forward(inputs_role, role_onehot)
            q_role_taken1 = q_vals1_role
            q_role_taken2 = q_vals2_role
            with th.no_grad():
                v_role1 = self.role_value(inputs_role).detach()
                # v_role1 = self.target_role_value(inputs_role).detach()
                v_role2 = v_role1
        else:
            q_vals1_role = self.role_critic1.forward(inputs_role)  # [..., n_roles]
            q_vals2_role = self.role_critic2.forward(inputs_role)  # [..., n_roles]
            q_role_taken1 = th.gather(q_vals1_role, dim=3, index=role).squeeze(3)
            q_role_taken2 = th.gather(q_vals2_role, dim=3, index=role).squeeze(3)

            v_role1 = th.logsumexp(q_vals1_role / alpha, dim=-1) * alpha
            v_role2 = th.logsumexp(q_vals2_role / alpha, dim=-1) * alpha

        # Get Q joint for roles (using individual Qs and Vs)
        q_vals1_role = self.role_mixer1(q_role_taken1, role_states, actions=role_onehot, vs=v_role1)
        q_vals2_role = self.role_mixer2(q_role_taken2, role_states, actions=role_onehot, vs=v_role2)

        return (q_vals1, q_vals2), (q_vals1_role, q_vals2_role)

    # self._get_q_values_no_grad(inputs[:, :-1], inputs_role, action_out, role_out)
    def _get_q_values_no_grad(self, inputs, inputs_role, action, role):
        """
        Get flattened individual Q values
        """
        with th.no_grad():
            # Get Q values
            if self.continuous_actions:
                action_input = action
                q_vals1 = self.critic1.forward(inputs, action_input)
                q_vals2 = self.critic2.forward(inputs, action_input)
                q_vals = th.min(q_vals1, q_vals2)
                q_vals = q_vals.view(-1)
            else:
                q_vals1 = self.critic1.forward(inputs)
                q_vals2 = self.critic2.forward(inputs)
                q_vals = th.min(q_vals1, q_vals2)
                q_vals = q_vals.view(-1, self.n_actions)

            if self.use_role_value:
                role_input = F.one_hot(role.squeeze(-1), num_classes=self.n_roles)
                q_vals1_role = self.role_critic1.forward(inputs_role, role_input)
                q_vals2_role = self.role_critic2.forward(inputs_role, role_input)
                q_vals_role = th.min(q_vals1_role, q_vals2_role)
                q_vals_role = q_vals_role.view(-1)
            else:
                q_vals1_role = self.role_critic1.forward(inputs_role)
                q_vals2_role = self.role_critic2.forward(inputs_role)
                q_vals_role = th.min(q_vals1_role, q_vals2_role)
                q_vals_role = q_vals_role.view(-1, self.n_roles)

        return q_vals, q_vals_role

    def train_encoder_continuous(self, batch, t_env, episode_num):
        pass
        
    def train_encoder_discrete(self, batch, t_env, episode_num):
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
            pred_r_loss = F.mse_loss(r_pred,  repeated_rewards)

            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimizer.zero_grad()
            pred_loss.backward()
            pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
            self.action_encoder_optimizer.step()

            if t_env > self.args.role_action_spaces_update_start:
                self.mac.update_role_action_spaces()
                self.role_action_spaces_updated = False
                self.n_roles = self.mac.n_roles
                # self._update_targets() # No target MAC
                
            self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
            self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
            self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm.item(), t_env)
            
    def train_decoder(self, batch, t_env):
        raise NotImplementedError

    def train_actor(self, batch, t_env, alpha):
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
        mask = mask.repeat(1, 1, self.n_agents)
        alpha: float = max(0.05, 0.5 - t_env / 200000)  # linear decay
        role_at = int(np.ceil((max_t - 1) / self.role_interval))  # always the same size as role_out
        role_t = role_at * self.role_interval
        role_mask = self._to_role_tensor(mask, role_t, max_t - 1)
        role_mask = role_mask.view(bs, role_at, self.role_interval, -1)[:, :, 0]
        role_mask = role_mask.reshape(-1)
        mask = mask.reshape(-1)

        avail_actions = batch["avail_actions"]

        # [ep_batch.batch_size, max_t, self.n_agents, -1]
        mac_out, mac_role_out = self._get_policy(batch, self.mac, avail_actions=avail_actions)
        
        role_taken, log_p_role = mac_role_out

        if self.use_role_value:
            log_p_role = log_p_role.reshape(-1)  # [-1]
            role_entropies = - (th.exp(log_p_role) * log_p_role).mean().item()
        else:
            log_p_role = log_p_role.reshape(-1, self.n_roles)  # [-1, self.n_roles]
            pi_role = th.exp(log_p_role)
            role_entropies = - (pi_role * log_p_role).sum(dim=-1).mean().item()

        # [batch.batch_size, max_t, self.n_agents]
        action_out, log_p_action = mac_out
        action_out = action_out[:, :-1]
        log_p_action = log_p_action[:, :-1]  # remove last step 

        if self.continuous_actions:
            log_p_action = log_p_action.reshape(-1)
            entropies = - (th.exp(log_p_action) * log_p_action).mean().item()
        else:
            log_p_action = log_p_action.reshape(-1, self.n_actions)
            pi = action_out.reshape(-1, self.n_actions)
            entropies = - (pi * log_p_action).sum(dim=-1).mean().item()

        # inputs are shared between v's and q's
        # Make sure that critic stacks actions from the inside
        inputs = self.critic1._build_inputs(batch, bs, max_t)
        inputs_role = self.role_critic1._build_inputs(batch, bs, max_t)

        if self.args.obs_role:
            role_role_taken_one_hot = F.one_hot(role_taken.squeeze(-1), num_classes = self.n_roles)
            t_role = min(max_t-1, role_t)
            shape_role_taken_one_hot = list(role_role_taken_one_hot.shape)
            shape_role_taken_one_hot[1] = max_t-1
            role_taken_one_hot = th.zeros(shape_role_taken_one_hot, device = self.device)
            role_taken_one_hot[:, :t_role] = role_role_taken_one_hot.repeat_interleave(self.role_interval,dim = 1)[:, :t_role]
            inputs = th.cat([inputs[:, :-1], role_taken_one_hot], dim = -1)

        # Get Q values with no grad and flattened
        q_vals, q_vals_role = self._get_q_values_no_grad(inputs, inputs_role, action_out, role_taken)

        if self.continuous_actions:
            # Get values for act (is not necessary, but it helps with stability)
            v_actions = self.value(inputs[:, :-1])  # inputs [BS, T-1, ...] --> Outputs: [BS*T-1] [BS*TRole, (None or N_roles)]
            v_actions = v_actions.reshape(-1)
            act_target = (alpha * log_p_action - q_vals)
            v_act_target =  F.mse_loss(v_actions, (q_vals - alpha * log_p_action).detach(),  reduction='none')
            v_act_loss = (v_act_target * mask).sum() / mask.sum()
        else:
            act_target = (pi * (alpha * log_p_action - q_vals)).sum(dim=-1)
            v_act_loss = 0
        # act_loss
        act_loss = (act_target * mask).sum() / mask.sum()

        # As roles are discrete we don't really need a value net as we can estimate V directly
        if self.use_role_value:
            # Move V towards Q
            v_role = self.role_value(inputs_role).reshape(-1)
            role_target = (alpha * log_p_role - q_vals_role)
            v_role_target = F.mse_loss(v_role, (q_vals_role - alpha * log_p_role).detach(),  reduction='none')
            v_role_loss = (v_role_target * role_mask).sum() / role_mask.sum()
            # print('v_role', v_role_loss)
        else:
            # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
            role_target = (pi_role * (alpha * log_p_role - q_vals_role)).sum(dim=-1)
            # The val net of roles isn't updated
            v_role_loss = 0

        role_loss = (role_target * role_mask).sum() / role_mask.sum()
        # print('role loss', role_loss)
        loss_policy = act_loss + role_loss
        # self.logger.console_logger.info(f"loss_policy {loss_policy}")
        if self.continuous_actions:
            kl_loss = self.mac.get_kl_loss()[:, :-1]
            masked_kl_loss = (kl_loss * mask).sum() / mask.sum()
            loss_policy += masked_kl_loss

        # Optimize policy
        # retain_graph = True if (self.use_role_value or self.continuous_actions) else False
        self.p_optimizer.zero_grad()
        loss_policy.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimizer.step()

        # If a value net exists, then optimize it
        if self.use_role_value or self.continuous_actions:
            loss_value = v_act_loss + v_role_loss
            self.val_optimizer.zero_grad()
            loss_value.backward()
            th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
            self.val_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("act_loss", act_loss.item(), t_env)
            self.logger.log_stat("role_loss", role_loss.item(), t_env)
            # self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            if self.use_role_value:
                self.logger.log_stat("v_role_loss", v_role_loss.item(), t_env)
            if self.continuous_actions:
                self.logger.log_stat("v_act_loss", v_act_loss.item(), t_env)

            self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("act_entropy", entropies, t_env)
            self.logger.log_stat("role_entropy", role_entropies, t_env)
            # self.logger.log_stat("min_log_p_role", (log_p_role*role_mask).min().detach().cpu().item(), t_env)
            self.log_stats_t = t_env

    def train_critic(self, batch, t_env, alpha):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        states = batch["state"]
        actions_taken = batch["actions"][:, :-1]
        roles_taken = batch["roles"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        roles_onehot = batch["roles_onehot"]
        out = self._build_role_rollout(rewards, states[:, :-1], roles_taken, roles_onehot[:, :-1], terminated, mask)                                                                                          
        role_rewards, role_states, roles, roles_roles_onehot, role_terminated, role_mask = out

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        inputs_role = self.role_critic1._build_inputs(batch, bs, max_t)
            
        with th.no_grad():
            # Sample roles according to current policy and get their log probabilities
            mac_out, role_out = self._get_policy(batch, self.mac, avail_actions=avail_actions)

        # select action
        # get log p of actions
        next_role, log_p_role = role_out

        # [batch.batch_size, max_t, self.n_agents]
        next_action_out, log_p_action = mac_out

        if self.continuous_actions:
            buff_action_one_hot = None
            next_action = next_action_out
            log_p_action_taken = log_p_action[:, 1:]
        else:
            buff_action_one_hot = batch["actions_onehot"][:, :-1] # buffer actions are pre-processed
            next_action = Categorical(next_action_out).sample().long().unsqueeze(3)
            log_p_action_taken = th.gather(log_p_action, dim=3, index=next_action).squeeze(3)[:, 1:]

        if self.use_role_value:
            log_p_role_taken = log_p_role[:, 1:]
        else:
            log_p_role_taken = th.gather(log_p_role, dim=3, index=next_role).squeeze(3)[:, 1:]
        
        # add selected role to critic inputs as observation
        next_role_role_onehot = F.one_hot(next_role.detach().squeeze(-1), num_classes = self.n_roles)
        if self.args.obs_role:
            # next_role_role_one_hot_rep [BS, T, n_agents, n_roles]
            next_role_onehot = th.zeros_like(roles_onehot, device = self.device)
            # repeat role taken and pad with zeros
            # next_role_role_one_hot_rep = next_role_role_onehot[..., None].expand(-1, -1, -1, -1, self.role_interval).transpose(1, 3).flatten(-2, -1).transpose(1, 3)
            # same as 
            next_role_role_one_hot_rep = next_role_role_onehot.squeeze(-1).repeat_interleave(self.role_interval, dim = 1)
            role_t = min(next_role_role_one_hot_rep.shape[1], next_role_onehot.shape[1])
            next_role_onehot[:, :role_t] = next_role_role_one_hot_rep[:, :role_t]
            target_inputs = th.cat([inputs, next_role_onehot], dim = -1)
        else:
            target_inputs = inputs

        # Find Q values of actions and roles according to current policy

        target_act_joint_q, target_role_joint_q = self._get_joint_q_target(target_inputs, inputs_role, states,
                                                                            role_states,next_action, next_role,
                                                                             next_role_role_onehot, alpha)

        # build_td_lambda_targets deals with moving the targets 1 step forward
        target_v_act = build_td_lambda_targets(rewards, terminated, mask, target_act_joint_q, self.n_agents,
                                               self.args.gamma,
                                               self.args.td_lambda)

        target_v_role = build_td_lambda_targets(role_rewards, role_terminated, role_mask, target_role_joint_q,
                                                self.n_agents, self.args.gamma,
                                                self.args.td_lambda)

        #  Eq 9 in FOP Paper
        targets_act = target_v_act - alpha * log_p_action_taken.mean(dim=-1, keepdim=True)
        targets_role = target_v_role - alpha * log_p_role_taken.mean(dim=-1, keepdim=True)
                # add selected role to critic inputs as observation

        if self.args.obs_role:
            inputs = th.cat([inputs, roles_onehot], dim = -1)

        # Find Q values of actions and roles taken in batch
        q_act_taken, q_role_taken = self._get_joint_q(inputs[:, :-1], inputs_role[:, :-1], states[:, :-1],
                                                      role_states[:, :-1], actions_taken, roles[:, :-1],
                                                      buff_action_one_hot, roles_roles_onehot[:, :-1], alpha)

        q1_act_taken, q2_act_taken = q_act_taken  # double q

        q1_role_taken, q2_role_taken = q_role_taken  # double q

        td_error1_role = q1_role_taken - targets_role.detach()
        td_error2_role = q2_role_taken - targets_role.detach()

        td_error1_act = q1_act_taken - targets_act.detach()
        td_error2_act = q2_act_taken - targets_act.detach()

        # 0-out the targets that came from padded data
        role_mask = role_mask[:, :-1].expand_as(td_error1_role)
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
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (q1_act_taken * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets_act * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

    def _update_targets(self):

        if getattr(self.args, "polyak_update", False):
            
            tau = getattr(self.args, "tau", 0.005)
            polyak_update(self.critic1.parameters, self.target_critic1.parameters, tau)
            polyak_update(self.critic2.parameters, self.target_critic2.parameters, tau)

            polyak_update(self.role_critic1.parameters, self.role_target_critic1.parameters, tau)
            polyak_update(self.role_critic2.parameters, self.role_target_critic2.parameters, tau)

            polyak_update(self.mixer1.parameters, self.target_mixer1.parameters, tau)
            polyak_update(self.mixer2.parameters, self.target_mixer2.parameters, tau)

            polyak_update(self.role_mixer1.parameters, self.role_target_mixer1.parameters, tau)
            polyak_update(self.role_mixer2.parameters, self.role_target_mixer2.parameters, tau)
        else:

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
        self.critic2.cuda()

        self.mixer1.cuda()
        self.mixer2.cuda()

        self.target_critic1.cuda()
        self.target_critic2.cuda()

        self.target_mixer1.cuda()
        self.target_mixer2.cuda()

        self.role_mixer1.cuda()
        self.role_mixer2.cuda()

        self.role_target_mixer1.cuda()
        self.role_target_mixer2.cuda()

        self.role_critic1.cuda()
        self.role_critic2.cuda()

        self.role_target_critic1.cuda()
        self.role_target_critic2.cuda()

        if self.continuous_actions:
            self.value.cuda()
        if self.use_role_value:
            self.role_value.cuda()
            
                
    def _to_role_tensor(self, tensor, role_t, T_max_1):
        """
        Create a tensor representing roles each time step, the output is padded to be of size role_t
        """
        tensor_shape = tensor.shape
        # self.logger.console_logger.info(f"tensor_shape {tensor_shape}")
        roles_shape = list(tensor_shape)
        roles_shape[1] = role_t

        tensor_out = th.zeros(roles_shape, dtype=tensor.dtype, device = self.device)
        tensor_out[:, :T_max_1] = tensor.detach().clone()

        return tensor_out

    def _build_role_rollout(self, rewards, states, roles_taken, roles_onehot, terminated, mask):
        """
        # role_out already missing last?
        # Use batch to build role inputs
        Input: Rewards [B, T-1], states [B, T], roles [B, T-1], terminated [B, T-1]
        Output: Roles [B, RoleT, role_interval], Roles States [B, RoleT, role_interval, -1], Roles Terminated [B, RoleT,
         role_interval]
        """
        roles_shape_o = roles_taken.shape  # bs, T-1, agents
        bs = roles_shape_o[0]  # batch size
        T_max_1 = roles_shape_o[1]  # T - 1

        # Get role transitions from batch # tensor[: ,::role_interval]
        role_at = int(np.ceil(T_max_1 / self.role_interval))  # always the same size as role_out
        role_t = role_at * self.role_interval


        roles  = roles_taken[:, :T_max_1][:, ::self.role_interval]
        # roles = self._to_role_tensor(roles_taken, role_t, T_max_1)
        # roles = roles.view(bs, role_at, self.role_interval, self.n_agents, -1)[:, :, 0]

        roles_roles_onehot = roles_onehot[:, :T_max_1][:, ::self.role_interval]
        # roles_roles_onehot = self._to_role_tensor(roles_onehot, role_t, T_max_1)
        # roles_roles_onehot = roles_roles_onehot.view(bs, role_at, self.role_interval, self.n_agents, -1)[:, :, 0]

        role_states = states[:, :T_max_1][:, ::self.role_interval]
        # role_states = self._to_role_tensor(states, role_t, T_max_1)
        # role_states = role_states.view(bs, role_at, self.role_interval, -1)[:, :, 0]

        # role_terminated
        role_terminated = self._to_role_tensor(terminated, role_t, T_max_1)
        role_terminated = role_terminated.view(bs, role_at, self.role_interval).sum(dim=-1, keepdim=True)

        # role_rewards
        role_rewards = self._to_role_tensor(rewards, role_t, T_max_1)
        role_rewards = role_rewards.view(bs, role_at, self.role_interval).sum(dim=-1, keepdim=True)

        # role_mask
        role_mask = mask[:, :T_max_1][:, ::self.role_interval]
        # role_mask = self._to_role_tensor(mask, role_t, T_max_1)
        # role_mask = role_mask.view(bs, role_at, self.role_interval, -1)[:, :, 0]

        return role_rewards, role_states, roles, roles_roles_onehot, role_terminated, role_mask
        
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

    def _build_ent_coeficient(self, args):
        # TODO: add auto alpha update
        if getattr(self.args, "alpha_auto", False):
            init_value = 1.0
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.alpha = (th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.RMSprop([self.alpha],  lr = args.lr, alpha = args)
            self.get_alpha = lambda t_env: self.alpha.detach()
            self.target_entropy = -np.prod(self.scheme.action_space.shape).astype(np.float32)
        else:
            self.get_alpha = lambda t_env: max(0.05, 0.5 - t_env/200000.0)

        # Default initial value of ent_coef when learned


