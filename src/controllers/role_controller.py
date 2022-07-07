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
# import numpy as np
import copy
import torch.nn.functional as F


# This multi-agent controller shares parameters between agents
class ROLEMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.continuous_actions = args.continuous_actions
        self.use_role_value = args.use_role_value
        self.role_interval = args.role_interval

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.n_roles = args.n_roles
        self._build_roles()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args) if not self.continuous_actions else None
        self.role_selector = role_selector_REGISTRY[args.role_selector](input_shape, args)
        self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)

        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None
        self.n_clusters = args.n_role_clusters

        self.role_latent = th.rand(self.n_roles, self.args.action_latent_dim).to(args.device)
        self.action_repr = th.rand(self.n_actions, self.args.action_latent_dim).to(args.device)

        self.role_action_spaces = None


        self.kl_loss = None


        self.actions_forward = self.continuos_actions_forward if self.continuous_actions else self.discrete_actions_forward
        self.logger = None
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Return valid actions (in action space)
        (agent_outputs, _), (_, _) = self.forward(ep_batch, t = t_ep, test_mode=test_mode, t_env=t_env)
        
        if self.continuous_actions:
            return agent_outputs, self.selected_roles
        else:
            avail_actions = ep_batch["avail_actions"][:, t_ep]
            # TODO: Implement action spaces (as in RODE)
            # TODO: Make log probs aware of role_avail_actions

            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
           
            return chosen_actions, self.selected_roles

    def forward(self, ep_batch, t, test_mode=False, t_env=None):
        # self.action_selector.logger = self.logger
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)

        agent_inputs = self._build_inputs(ep_batch, t) 
        batch_size = ep_batch.batch_size

        role_outputs = None
        log_p_role = None
        # select a role every self.role_interval steps
        if t % self.role_interval == 0:
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)

            # Get Index of the role of each agent
            selected_roles, log_p_role = self.role_selector.select_role(role_outputs, test_mode=test_mode,
                                                                        t_env=t_env)

            self.selected_roles = selected_roles

            selected_roles = selected_roles.unsqueeze(-1).view(batch_size, self.n_agents, -1)


            if self.use_role_value:
                role_outputs = selected_roles
                log_p_role = log_p_role.view(batch_size, self.n_agents)
            else:
                role_outputs = self.softmax_roles(role_outputs, batch_size, test_mode) 
                log_p_role = None
                
        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        (actions, log_p_action)  = self.actions_forward(batch_size, avail_actions, t_env, test_mode)

        return (actions, log_p_action), (role_outputs, log_p_role)

    def softmax_roles(self, role_outs, batch_size, test_mode):


        role_outs = F.softmax(role_outs, dim=-1)

        if not test_mode:
            # Epsilon floor
            epsilon_action_num = role_outs.size(-1)

            role_outs = ((1 - self.action_selector.epsilon) * role_outs
                            + th.ones_like(role_outs) * self.role_selector.epsilon/epsilon_action_num)

        return role_outs.view(batch_size, self.n_agents, -1)

    def softmax_actions(self, agent_outs, batch_size, avail_actions, test_mode):
        # Asumes that action selector is multinomial
        if getattr(self.args, "mask_before_softmax", True):
            # Make the logits for unavailable actions very negative to minimise their affect on the softmax
            reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
            agent_outs[avail_actions == 0] = -1e11

        agent_outs = F.softmax(agent_outs, dim=-1)
        if not test_mode:
            # Epsilon floor
            epsilon_action_num = agent_outs.size(-1)
            if getattr(self.args, "mask_before_softmax", True):
                # With probability epsilon, we will pick an available action uniformly
                epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

            agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                            + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

            if getattr(self.args, "mask_before_softmax", True):
                # Zero out the unavailable actions
                agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(batch_size, self.n_agents, -1)

    def continuos_actions_forward(self, batch_size, avail_actions, t_env, test_mode):
        actions, log_p_action, kl_loss = [], [], []
        for role_i in range(self.n_roles):
            # [bs * n_agents, n_actions]
            dist_params = self.roles[role_i](self.hidden_states, self.action_repr)
            prior = self.roles[role_i].prior
            pi_action, log_p_action_taken, dkl_loss = self.action_selector(*dist_params, prior = prior, test_mode = test_mode)

            actions.append(pi_action)
            log_p_action.append(log_p_action_taken)
            kl_loss.append(dkl_loss)


        kl_loss = th.stack(kl_loss, dim= -1)  # [bs*n_agents, n_roles]
        kl_loss = kl_loss.view(batch_size*self.n_agents, -1)
        kl_loss = kl_loss.gather(index = self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim = 1)
        kl_loss = kl_loss[: ,0]
        kl_loss = kl_loss.view(batch_size, self.n_agents)
        self.kl_loss = kl_loss

        log_p_action = th.stack(log_p_action, dim= -1)  # [bs*n_agents, n_roles]
        log_p_action = log_p_action.view(batch_size*self.n_agents, -1) 
        log_p_action = log_p_action.gather(index = self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim = 1)
        log_p_action = log_p_action[: ,0]
        log_p_action = log_p_action.view(batch_size, self.n_agents)  # [bs,n_agents]     

        actions = th.stack(actions, dim= -1)  # [bs*n_agents, dim_actions, n_roles]
        actions = actions.view(batch_size*self.n_agents, self.n_actions, -1)
        actions = actions.gather(index = self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim = -1)
        actions = actions[: ,0]
        actions = actions.view(batch_size, self.n_agents, self.n_actions, -1)

        return actions, log_p_action

    def discrete_actions_forward(self, batch_size, avail_actions, t_env, test_mode):
        pi = []
        for role_i in range(self.n_roles):
            pi_out = self.roles[role_i](self.hidden_states, self.action_repr)
            pi_out = pi_out.view(batch_size, self.n_agents, self.n_actions)
            pi.append(pi_out)

        pi = th.stack(pi, dim= -1)  # [batch_size, self.n_agents, self.n_actions, n_roles]
        pi = pi.view(batch_size*self.n_agents, self.n_actions, -1) # [batch_size*self.n_agents*self.n_actions, n_roles]
        pi = pi.gather(index = self.selected_roles.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_actions, self.n_roles), dim = -1)
        pi = pi[...,0]

        if self.agent_output_type == "pi_logits":
            pi = self.softmax_actions(pi, batch_size, avail_actions, test_mode)

        pi = pi.view(batch_size, self.n_agents, self.n_actions)

        return pi, None

    def update_prior(self, role_i, mu, sigma):
        prior = Normal(mu, sigma)
        self.roles[role_i].update_prior(prior)

    def update_decoder(self, decoder):
        self.action_selector.update_decoder(decoder)

    def get_kl_loss(self):
        return self.kl_loss

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.role_hidden_states = self.role_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents,
                                                                                    -1)  # bav

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

    def _build_agents(self, input_shape):
        # Politica de agente individual
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        # Politica de selector de roles
        self.role_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_roles(self):
        self.roles = [role_REGISTRY[self.args.role](self.args) for _ in range(self.n_roles)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t], th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)]
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        # Add agent ID to input
        input_shape += self.n_agents

        return input_shape

    def update_roles(self):
        action_repr = self.action_encoder()
        raise NotImplementedError

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])
