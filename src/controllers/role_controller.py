# from tkinter import N
import pip
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from modules.roles import REGISTRY as role_REGISTRY
from components.role_selectors import REGISTRY as role_selector_REGISTRY
from torch.distributions.normal import Normal
import torch as th
import numpy as np
from itertools import cycle

# import numpy as np
import copy
import torch.nn.functional as F
from sklearn.cluster import KMeans

# This multi-agent controller shares parameters between agents
class ROLEMAC:
    def __init__(self, scheme, groups, args):

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.continuous_actions = args.continuous_actions
        self.use_role_value = args.use_role_value
        self.role_interval = args.role_interval
        self.n_roles = args.n_roles
        self.n_clusters = args.n_role_clusters 
        self.agent_output_type = args.agent_output_type
        self.shared_encoder = getattr(args, 'shared_encoder', False) # share prediction model and policy action representations
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self._build_roles()

        # Selectors
        self.action_selector = action_REGISTRY[args.action_selector](args) if not self.continuous_actions else None
        self.role_selector = role_selector_REGISTRY[args.role_selector](input_shape, args)
        self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)

        # Temp variables 
        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None

        # Role latent and actions representations
        self.role_latent = th.ones(self.n_roles, self.args.action_latent_dim).to(args.device)
        
        if not self.continuous_actions:
            self.forward = self._discrete_forward
            self.select_actions = self._select_actions_discrete 
            self.action_repr = th.from_numpy(np.random.rand(self.n_actions, self.args.action_latent_dim)).float().to(args.device)
            self.role_action_spaces = th.ones(self.n_roles, self.n_actions).to(args.device)
            self.action_spaces_updated = False
        else:
            self.mask_before_softmax = args.mask_before_softmax
            self.forward = self._continuous_forward
            self.select_actions = self._select_actions_continuous 
            self.kl_loss = None

        if getattr(self.args, 'relabeling', False):
            self.update_pi_buffer = self._buffer_pi
        else:
            self.update_pi_buffer = self._no_buffer_pi


        if getattr(self.args, 'shared_encoder', False):
            self.encoded_actions = self._from_encoder
        else:
            self.encoded_actions = self._from_frozen
        
    def _from_frozen(self):
        return self.action_repr

    def _from_encoder(self):
        with th.no_grad():
            return self.action_encoder()

    def _select_actions_continuous(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        with th.no_grad():
            (agent_outputs, _), (_, _) = self.forward(ep_batch, t=t_ep, test_mode=test_mode, t_env=t_env, explore = True)
        selected_roles = self.selected_roles.view(ep_batch.batch_size, self.n_agents, -1)     
        return agent_outputs[bs].detach(), selected_roles[bs].detach()

    def _select_actions_discrete(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        # Chose a role and then an action, (agent_outputs are masked)
        with th.no_grad():
            (agent_outputs, _), (_, _) = self.forward(ep_batch, t=t_ep, test_mode=test_mode, t_env=t_env)
            
            avail_actions = ep_batch["avail_actions"][:, t_ep]
            
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                                test_mode=test_mode)
                                                                
            selected_roles = self.selected_roles.view(ep_batch.batch_size, self.n_agents, -1)  

        return chosen_actions.detach(), selected_roles[bs].detach()

    def _get_avail_actions_role(self, batch_size):
        # self.selected_roles [BS*n_agents]
        role_index= self.selected_roles.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_actions).long()
        role_avail_actions = th.gather(self.role_action_spaces.unsqueeze(0).expand(batch_size * self.n_agents, -1, -1), dim=1, index= role_index)
        role_avail_actions = role_avail_actions[:, 0] 
        self.role_avail_actions = role_avail_actions.view(batch_size, self.n_agents, -1)
        
        return role_avail_actions.view(batch_size, self.n_agents, -1)

    def _discrete_forward(self, ep_batch, t, test_mode=False, t_env=None, explore = True):
        
        # self.action_selector.logger = self.logger
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)

        selected_roles = None
        log_p_role = None
        # select a role every self.role_interval steps
        if t % self.role_interval == 0:
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)
            role_pis =  self._softmax_roles(role_outputs, batch_size, test_mode = test_mode, explore = explore)
            # Get Index of the role of each agent
            selected_roles, log_p_role = self.role_selector.select_role(role_pis, test_mode=test_mode,
                                                                        t_env=t_env)
            
            self.selected_roles = selected_roles

            self.role_avail_actions = self._get_avail_actions_role(batch_size)
            
            selected_roles = self.selected_roles.unsqueeze(-1).view(batch_size, self.n_agents, -1)

            log_p_role = th.log(role_pis.view(batch_size, self.n_agents, -1))

        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        (pi_action, log_p_action) = self._discrete_actions_forward(batch_size, avail_actions, t_env, test_mode, explore)

        return (pi_action, log_p_action), (selected_roles, log_p_role)

    def _continuous_forward(self, ep_batch, t, test_mode=False, t_env=None, explore = True):

        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size
        
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)

        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size

        selected_roles = None
        log_p_role = None
        # select a role every self.role_interval steps
        if t % self.role_interval == 0:
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)
            role_pis =  self.softmax_roles(role_outputs, batch_size, test_mode = test_mode)
            # Get Index of the role of each agent
            selected_roles, log_p_role = self.role_selector.select_role(role_pis, test_mode=test_mode,
                                                                        t_env=t_env)
            self.selected_roles = selected_roles.squeeze(-1)
            selected_roles = selected_roles.unsqueeze(-1).view(batch_size, self.n_agents, -1)

            log_p_role = th.log(role_pis.view(batch_size, self.n_agents, -1))

        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        (pi_action, log_p_action) = self._continuos_actions_forward(batch_size, avail_actions, t_env, test_mode, explore)

        return (pi_action, log_p_action), (selected_roles, log_p_role)


    def _continuos_actions_forward(self, batch_size, avail_actions, t_env, test_mode, explore):

        actions, log_p_action, kl_loss = [], [], []
        
        for role_i in range(self.n_roles):
            # [bs * n_agents, n_actions]
            dist_params = self.roles[role_i](self.hidden_states)
            prior = self.roles[role_i].prior
            pi_action, log_p_action_taken, dkl_loss = self.action_selector(*dist_params, prior=prior,
                                                                           test_mode=test_mode, t_env = t_env)
            
            actions.append(pi_action)
            log_p_action.append(log_p_action_taken)
            kl_loss.append(dkl_loss)

        # Create a buffer for relabeling (only when relabeling is enabled)
        self.update_pi_buffer(log_p_action_taken) # [batch_size, self.n_agents, self.n_actions, n_roles]

        kl_loss = th.stack(kl_loss, dim=-1)  # [bs*n_agents, n_roles]
        kl_loss = kl_loss.view(batch_size * self.n_agents, -1)
        kl_loss = kl_loss.gather(index=self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim=1)
        kl_loss = kl_loss[:, 0]
        kl_loss = kl_loss.view(batch_size, self.n_agents)
        self.kl_loss = kl_loss

        log_p_action = th.stack(log_p_action, dim=-1)  # [bs*n_agents, n_roles]
        log_p_action = log_p_action.view(batch_size * self.n_agents, -1)
        log_p_action = log_p_action.gather(index=self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim=1)
        log_p_action = log_p_action[:, 0]
        log_p_action = log_p_action.view(batch_size, self.n_agents)  # [bs,n_agents]     

        actions = th.stack(actions, dim=-1)  # [bs*n_agents, dim_actions, n_roles]
        actions = actions.view(batch_size * self.n_agents, self.n_actions, -1)
        actions = actions.gather(index=self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim=-1)
        actions = actions[:, 0]
        actions = actions.view(batch_size, self.n_agents, self.n_actions, -1)
        
        return actions, log_p_action

    def _discrete_actions_forward(self, batch_size, avail_actions, t_env, test_mode, explore):

        pi = []
        for role_i in range(self.n_roles):
            pi_out = self.roles[role_i](self.hidden_states, self.encoded_actions())
            pi_out = pi_out.view(batch_size, self.n_agents, self.n_actions)
            pi.append(pi_out)

        pi = th.stack(pi, dim=-1)  

        # Create a buffer for relabeling (only when relabeling is enabled)
        self.update_pi_buffer(pi, t_env) # [batch_size, self.n_agents, self.n_actions, n_roles]

        pi = pi.view(batch_size * self.n_agents, self.n_actions,
                     -1)  # [batch_size*self.n_agents*self.n_actions, n_roles]
        pi = pi.gather(index=self.selected_roles.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_actions, self.n_roles),
                       dim=-1)
        pi = pi[..., 0]

        if self.agent_output_type == "pi_logits":
            pi = self._softmax_actions(pi, batch_size, avail_actions, test_mode, explore)

        pi = pi.view(batch_size, self.n_agents, -1)

        return pi, None

    def _softmax_roles(self, role_outs, batch_size, test_mode, explore = True):

        role_outs = F.softmax(role_outs, dim=-1)

        if (not test_mode) and explore:
            # Epsilon floor
            epsilon_action_num = role_outs.size(-1)
            
            role_outs = ((1 - self.action_selector.epsilon) * role_outs
                         + th.ones_like(role_outs) * self.role_selector.epsilon / epsilon_action_num)

        return role_outs

    def _softmax_actions(self, agent_outs, batch_size, avail_actions, test_mode, explore = True):
        # Apply role mask (is applied before softmax to avoid no action available)
        role_avail_actions = self.role_avail_actions.reshape(batch_size * self.n_agents, -1)
        agent_outs[role_avail_actions == 0] = -1e11

        # Apply mask and softmax
        if getattr(self.args, "mask_before_softmax", True):
            # Make the logits for unavailable actions very negative to minimize their affect on the softmax
            reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
            
            agent_outs[reshaped_avail_actions == 0] = -1e11

        agent_outs = F.softmax(agent_outs, dim=-1)

        if (not test_mode) and explore:
            # Epsilon floor
            epsilon_action_num = agent_outs.size(-1)
            if getattr(self.args, "mask_before_softmax", True):
                # With probability epsilon, we will pick an available action uniformly
                epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

            agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                          + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

            if getattr(self.args, "mask_before_softmax", True):
                # Zero out the unavailable actions
                agent_outs[reshaped_avail_actions == 0] = 0.0
            
        return agent_outs.view(batch_size, self.n_agents, -1)

    def get_role_probs_discrete(self, ep_batch, t, t_env = None, test_mode = False, explore = True):
        batch_size = ep_batch.batch_size
        avail_actions = ep_batch["avail_actions"][:, t]
        old_action = ep_batch["actions"][:, t]
        old_action = old_action.reshape(batch_size * self.n_agents, 1 , -1)

        pi = self.pi_buffer
        pi = pi.transpose(-1, -2).view(batch_size * self.n_agents, self.n_roles, self.n_actions)  # [batch_size* n_agents, n_roles, n_actions]

        # [batch_size* n_agents, n_actions, n_roles]
        role_avail_actions = self.role_action_spaces.unsqueeze(0).expand(batch_size * self.n_agents, -1, -1)

        pi[role_avail_actions == 0] = -1e11

        # Apply mask and softmax
        if getattr(self.args, "mask_before_softmax", True):
            # avail_actions [bs, n_agents, n_actions, n_roles]
            v_avail_actions = avail_actions.reshape(batch_size * self.n_agents, self.n_actions)
            v_avail_actions = v_avail_actions.unsqueeze(1).expand(batch_size * self.n_agents, self.n_roles, self.n_actions)
            # Make the logits for unavailable actions very negative to minimize their affect on the softmax
            pi[v_avail_actions == 0] = -1e11
        
        pi = F.softmax(pi, dim= -1) # softmax over actions

        if (not test_mode) and explore:
            # Epsilon floor
            epsilon_action_num = pi.size(-1)

            if getattr(self.args, "mask_before_softmax", True):
                # With probability epsilon, we will pick an available action uniformly
                epsilon_action_num = v_avail_actions.sum(dim= -1, keepdim=True).float()

            pi = ((1 - self.action_selector.epsilon) * pi
                          + th.ones_like(pi) * self.action_selector.epsilon / epsilon_action_num)

            if getattr(self.args, "mask_before_softmax", True):
                # Zero out the unavailable actions
                pi[v_avail_actions == 0] = 0.0  
        
        # old_action -> bs, n_agents, -1, -1
        pi_old = th.gather(pi, index = old_action.expand(-1, self.n_roles, self.n_actions), dim = -1)
        pi_old = pi_old[..., 0]

        return pi_old.view(batch_size, self.n_agents, self.n_roles)

    def _buffer_pi(self, pi, t_env):
        # only buffer during training (t_env is None)
        if t_env == None:
            self.pi_buffer = pi.detach() # .clone() 

    def _no_buffer_pi(self, pi, t_env): 
        pass

    def update_prior(self, role_i, mu, sigma):
        prior = Normal(mu, sigma)
        self.roles[role_i].update_prior(prior)

    def update_decoder(self, decoder):
        self.action_selector.update_decoder(decoder)

    def get_kl_loss(self):
        return self.kl_loss

    def init_hidden(self, batch_size):
        self.hidden_states = None
        self.role_hidden_states = None

    def parameters(self):
        params = list(self.agent.parameters())
        params += list(self.role_agent.parameters())
        for role_i in range(self.n_roles):
            params += list(self.roles[role_i].parameters())
        params += list(self.role_selector.parameters())

        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.role_agent.load_state_dict(other_mac.role_agent.state_dict())
        if other_mac.n_roles > self.n_roles:
            self.n_roles = other_mac.n_roles
            self.roles = copy.deepcopy(other_mac.roles)
        else:
            for role_i in range(self.n_roles):
                self.roles[role_i].load_state_dict(other_mac.roles[role_i].state_dict())

        self.role_selector.load_state_dict(other_mac.role_selector.state_dict())
        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.role_latent = copy.deepcopy(other_mac.role_latent)
        self.action_repr = copy.deepcopy(other_mac.action_repr)

    def _build_agents(self, input_shape):
        # agent w/o roles policy 
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        # rol selector policy
        self.role_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_roles(self):
        self.roles = [role_REGISTRY[self.args.role](self.args) for _ in range(self.n_roles)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        print(input_shape)
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        print(input_shape)
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        print(input_shape)
        return input_shape

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])

    def update_role_action_spaces(self):
        """
        (Discrete)
        Action spaces from rode 
        https://github.com/TonghanWang/RODE
        (SC2 Only)
        """

        action_repr = self.action_encoder()
        action_repr_array = action_repr.detach().cpu().numpy()  # [n_actions, action_latent_d]

        n_roles = 1
        add_clusters = 0
        while n_roles < self.n_roles:
            print(self.n_clusters + add_clusters, n_roles)
            k_means = KMeans(n_clusters=self.n_clusters + add_clusters, random_state=0).fit(action_repr_array)
            spaces = []
            for cluster_i in range(self.n_clusters):
                spaces.append((k_means.labels_ == cluster_i).astype(np.float))

            o_spaces = copy.deepcopy(spaces)
            spaces = []

            for space_i ,space in enumerate(o_spaces):
                _space = copy.deepcopy(space)
                _space[0] = 0.
                _space[1] = 0.

                # Add outliers to every tole
                if _space.sum() <= 1.:
                    o_spaces = np.minimum(o_spaces[space_i] + o_spaces, 1.0)   
                elif _space[:6].sum() >= 2. and _space[6:].sum() >= 1.: 
                    spaces.append(o_spaces[space_i])
                else:
                    _space[:6] = 1.
                    spaces.append(_space)

            # Allow idling while dead
            for space in spaces:
                space[0] = 1.

            spaces = np.unique(spaces, axis=0)
            spaces = list(spaces)
            n_roles = len(spaces)

            if n_roles < self.n_roles:
                spaces.append(np.ones_like(spaces[0]))
            n_roles = len(spaces)  

            add_clusters += 1
            
            
        cyclic_spaces = cycle(spaces)
        
        while n_roles < self.n_roles:
            spaces.append(next(cyclic_spaces)) 
            n_roles += 1

        # use role latent as the critic input and make role Q output 1 dimensional 
        expandable = (self.use_role_value and getattr(self.args, 'use_role_latent', False))

        if n_roles > self.n_roles:
            if not expandable:
                # merge clusters
                sorted_spaces = sorted(spaces, key = lambda x: -x.sum())
                cyclic_roles = cycle(reversed(range(self.n_roles))) # cycle from smaller to bigger spaces
                for small_space in sorted_spaces[self.n_roles:]:
                    i = next(cyclic_roles)
                    # combine the smallest spaces
                    space_comb = np.minimum(sorted_spaces[i] + small_space, 1)  
                    sorted_spaces[i] = space_comb
                spaces = sorted_spaces[:self.n_roles]
            else:
                # add roles
                if n_roles > self.n_roles:
                    for _ in range(self.n_roles, n_roles):
                        self.roles.append(role_REGISTRY[self.args.role](self.args))
                        if self.args.use_cuda:
                            self.roles[-1].cuda()
                # Sort spaces by length (make existing roles be used for bigger spaces)
                spaces = sorted_spaces = sorted(spaces, key = lambda x: -x.sum())
        n_roles = len(spaces)
        self.n_roles = n_roles 

        print('>>> Role Action Spaces', spaces)

        for role_i, space in enumerate(spaces):
            self.roles[role_i].update_action_space(space)

        self.role_action_spaces = th.Tensor(np.array(spaces)).to(self.args.device).float()  # [n_roles, n_actions]
        
        self.role_latent = th.matmul(self.role_action_spaces, action_repr) / self.role_action_spaces.sum(dim=-1,
                                                                                                         keepdim=True)
        self.role_latent = self.role_latent.detach().clone()
        self.action_repr = action_repr.detach().clone()
        self.encoded_actions = self._from_frozen
        print(th.max(self.action_repr), th.min(self.action_repr))
        print(th.max(self.role_latent), th.min(self.role_latent))
        print(th.max(self.encoded_actions()), th.min(self.encoded_actions()))

    def cuda(self):
        self.agent.cuda()
        self.role_agent.cuda()
        for role_i in range(self.n_roles):
            self.roles[role_i].cuda()
        self.role_selector.cuda()
        self.action_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}/agent.th")
        th.save(self.role_agent.state_dict(), f"{path}/role_agent.th")
        for role_i in range(self.n_roles):
            th.save(self.roles[role_i].state_dict(), f"{path}/role_{role_i}.th")
        th.save(self.role_selector.state_dict(), f"{path}/role_selector.th")

        th.save(self.action_encoder.state_dict(), f"{path}/action_encoder.th")
        th.save(self.role_action_spaces, f"{path}/role_action_spaces.pt")
        th.save(self.role_latent, f"{path}/role_latent.pt")
        th.save(self.action_repr, f"{path}/action_repr.pt")

    def load_models(self, path):
        self.n_roles = self.role_action_spaces.shape[0]
        self.agent.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
        self.role_agent.load_state_dict(
            th.load(f"{path}/role_agent.th", map_location=lambda storage, loc: storage))
        for role_i in range(self.n_roles):
            try:
                self.roles[role_i].load_state_dict(th.load(f"{path}/role_{role_i}.th",
                                                           map_location=lambda storage, loc: storage))
            except:
                self.roles.append(role_REGISTRY[self.args.role](self.args))
            if self.args.use_cuda:
                self.roles[role_i].cuda()

        self.role_selector.load_state_dict(th.load(f"{path}/role_selector.th",
                                                   map_location=lambda storage, loc: storage))

        self.action_encoder.load_state_dict(th.load(f"{path}/action_encoder.th",
                                                    map_location=lambda storage, loc: storage))
        
        self.role_latent = th.load(f"{path}/role_latent.pt",
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        self.action_repr = th.load(f"{path}/action_repr.pt",
                                   map_location=lambda storage, loc: storage).to(self.args.device)
