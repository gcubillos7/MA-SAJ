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

        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters()) + list(self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())
        
        self.value = ValueNet(args)
        if args.use_role_value:
            self.role_value = ValueNet(args)
            self.value_params = list(self.value.parameters()) + list(self.role_value.parameters())
        else:
            self.value_params =  list(self.value.parameters())

        self.agent_params = list(mac.parameters())

        # Use FOP mixer
        self.role_mixer1 = FOPMixer(args)
        self.role_mixer2 = FOPMixer(args)

        self.role_target_mixer1 = copy.deepcopy(self.role_mixer1)
        self.role_target_mixer2 = copy.deepcopy(self.role_mixer2)

        # Use rnn + dot product --> Q, V using definition
        self.role_critic1 = FOPCritic(scheme, args)
        self.role_critic2 = FOPCritic(scheme, args)

        self.role_target_critic1 = copy.deepcopy(self.role_critic1)
        self.role_target_critic2 = copy.deepcopy(self.role_critic2)

        self.role_critic_params1 = list(self.role_critic1.parameters()) + list(self.role_mixer1.parameters())
        self.role_critic_params2 = list(self.role_critic2.parameters()) + list(self.role_mixer2.parameters())

        self.p_optimiser = RMSprop(params= self.agent_params , lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        self.c_optimiser1 = RMSprop(params=self.critic_params1 + self.role_critic_params1, lr=args.c_lr, alpha=args.optim_alpha,
                                    eps=args.optim_eps)

        self.c_optimiser2 = RMSprop(params=self.critic_params2 + self.role_critic_params2, lr=args.c_lr, alpha=args.optim_alpha,
                                    eps=args.optim_eps)

        self.val_optimiser = RMSprop(params=self.value_params, lr=args.v_lr, alpha=args.optim_alpha,
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

    def _get_joint_q_target(self, target_inputs, pi_a, pi_role):


        target_q_vals1 = self.target_critic1.forward(target_inputs).detach()
        target_q_vals2 = self.target_critic2.forward(target_inputs).detach()

        # if not use_role_value
        # # directly caculate the values by definition
        # next_vs1 = th.logsumexp(target_q_vals1 / alpha, dim=-1) * alpha
        # next_vs2 = th.logsumexp(target_q_vals2 / alpha, dim=-1) * alpha
        
        next_chosen_qvals1 = th.gather(target_q_vals1, dim=3, index=next_actions).squeeze(3)
        next_chosen_qvals2 = th.gather(target_q_vals2, dim=3, index=next_actions).squeeze(3)
        with th.no_grad:
            value = 

        target_qvals1 = self.target_mixer1(next_chosen_qvals1, states, actions=next_actions_onehot, vs=next_vs1)
        target_qvals2 = self.target_mixer2(next_chosen_qvals2, states, actions=next_actions_onehot, vs=next_vs2)
        
        pass

    def _get_joint_q(self, inputs, pi_a, pi_role):

        # Get Q values
        q_vals1 = self.critic1.forward(inputs, pi_a) 
        q_vals2 = self.critic2.forward(inputs, pi_a)
        
        
        if self.args.use_role_value:
            #inputs = self.role_critic1._build_inputs(batch, bs, max_t)
            q_vals1_role = self.role_critic1.forward(inputs, pi_role)
            q_vals2_role = self.role_critic2.forward(inputs, pi_role)
            q_vals_role = th.min(q_vals1_role, q_vals2_role) # Q(T,a)
            q_vals_role = q_vals_role[:, :-1]
            q_vals_role = q_vals_role.reshape(-1)
        else:  

            q_vals1_role = self.role_critic1.forward(inputs) 
            q_vals2_role = self.role_critic2.forward(inputs)

            
            q_vals_role = th.min(q_vals1_role, q_vals2_role) # Q(T) -> q(1),q(2), ...,q(n_roles)
            q_vals_role = q_vals_role[:, :-1]
            q_vals_role = q_vals_role.reshape(-1, self.mac.n_roles)

        return q_vals, q_vals_role

    def _get_q_values_no_grad(self, inputs, pi_a, pi_role):

        with th.no_grad():
            # Get Q values
            q_vals1 = self.critic1.forward(inputs, pi_a) 
            q_vals2 = self.critic2.forward(inputs, pi_a)
            q_vals = th.min(q_vals1, q_vals2) # Q(T,a) -> Q [Batch_size, seq, n_agents]
            q_vals = q_vals[:, :-1]
            q_vals = q_vals.reshape(-1)

            if self.args.use_role_value:
                #inputs = self.role_critic1._build_inputs(batch, bs, max_t)
                q_vals1_role = self.role_critic1.forward(inputs, pi_role)
                q_vals2_role = self.role_critic2.forward(inputs, pi_role)
                q_vals_role = th.min(q_vals1_role, q_vals2_role) # Q(T,a)
                q_vals_role = q_vals_role[:, :-1]
                q_vals_role = q_vals_role.reshape(-1)
            else:
                q_vals1_role = self.role_critic1.forward(inputs) 
                q_vals2_role = self.role_critic2.forward(inputs)
                q_vals_role = th.min(q_vals1_role, q_vals2_role) # Q(T) -> q(1),q(2), ...,q(n_roles)
                q_vals_role = q_vals_role[:, :-1]
                q_vals_role = q_vals_role.reshape(-1, self.mac.n_roles)

        return q_vals, q_vals_role

    def train_actor(self, batch, t_env):
        """
        Update actor and value nets as in SAC (haarjona)
        https://github.com/haarnoja/sac/blob/master/sac/algos/sac.py  
        Add regularization term for implicit constraints 
        Mixer isn't used during policy improvement
        """
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        states = batch["state"]

        alpha: float = max(0.05, 0.5 - t_env / 200000)  # linear decay
        # [ep_batch.batch_size, max_t, self.n_agents, -1]
        mac_out, role_out = self.get_policy(batch, self.mac)
        
        # t_mac_out, t_role_out = self.get_policy(batch, self.target_mac)

        # select action
        # get log p of actions
        pi_role, log_pi_role = role_out

        # [batch.batch_size, max_t, self.n_agents]
        pi_a, log_pi_a = mac_out

        log_pi_a = log_pi_a[:, :-1].clone() # remove end 
        log_pi_a = log_pi_a.reshape(-1)  
        pi_a = pi_a[:, :-1]
        
        # inputs are shared between v's and q's
        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals, q_vals_role = self._get_q_values_no_grad(inputs, pi_a, pi_role)

        # Get values for act (is not necessary but it helps with stability)
        v_actions = self.values(inputs)
        v_actions = v_actions.reshape(-1)
        entropies = - (th.exp(log_pi_a) * log_pi_a).sum(dim=-1)

        act_target = (alpha * log_pi_a - q_vals).sum(dim=-1)
        v_act_target = (0.5 * (v_actions - (q_vals - alpha * log_pi_a) )**2).sum(dim = -1)

        # As is Discrete we don't really need a value net as we can estimate V directly
        if self.args.use_role_value:
            # Shape is flattened
            log_pi_role = log_pi_role[:, :-1]
            log_pi_role = log_pi_role.reshape(-1) 
            pi_role = pi_role[:, :-1]
            # Move V towards Q
            v_role = self.role_values(inputs).reshape(-1)
            v_role = v_role.reshape(-1)
            rol_target = (alpha * log_pi_role - q_vals_role).sum(dim=-1)
            v_role_target = (0.5 * (v_role - (q_vals_role - alpha* log_pi_role) )**2).sum(dim = -1)
            val_target = v_act_target + v_role_target
        else:
            # Get Q for each role
            log_pi_role_copy = log_pi_role[:, :-1].clone()
            log_pi_role_copy = log_pi_role_copy.reshape(-1, self.mac.n_roles)
            pi_role = th.exp(log_pi_role_copy) # Get p instead of log_p
            log_pi_role = log_pi_role[:, :-1].reshape(-1, self.mac.n_roles)
            # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
            rol_target = (pi_role * (alpha * log_pi_role - q_vals_role[:, :-1].reshape(-1, self.n_roles))).sum(dim=-1)
            # The val net of roles isn't updated
            val_target = v_act_target

        pol_target = act_target + rol_target       
             
        loss_policy = (pol_target * mask).sum() / mask.sum()
        loss_value = (val_target * mask).sum() / mask.sum()

        # Optimize values
        self.val_optimiser.zero_grad()
        loss_policy.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
        self.val_optimiser.step()

        # Optimize policy
        self.p_optimiser.zero_grad()
        loss_policy.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            # self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train_critic(self, batch, t_env):
        raise NotImplementedError

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