from tkinter import N
from src.modules.agents import REGISTRY as agent_REGISTRY
from src.components.action_selectors import REGISTRY as action_REGISTRY
from src.modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from src.modules.roles import REGISTRY as role_REGISTRY
from src.modules.role_selectors import REGISTRY as role_selector_REGISTRY
import torch as th

import numpy as np
import copy


# This multi-agent controller shares parameters between agents
class ROLEMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.role_interval = args.role_interval

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.n_roles = 3
        self._build_roles()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.role_selector = role_selector_REGISTRY[args.role_selector](input_shape, args)
        self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)

        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None
        self.n_clusters = args.n_role_clusters

        self.role_latent = th.ones(self.n_roles, self.args.action_latent_dim).to(args.device)
        self.action_repr = th.ones(self.n_actions, self.args.action_latent_dim).to(args.device)

    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Return valid actions (in action space)
        agent_outputs, role_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, t_env=t_env)

        return agent_outputs, role_outputs

    def forward(self, ep_batch, t, test_mode=False, t_env=None):

        agent_inputs = self._build_inputs(ep_batch, t)

        # select roles
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)
        role_outputs = None

        # select a role every self.role_interval steps
        if t % self.role_interval == 0:
            # Q value for each agent for each role 
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)
            # Get Index of the role of each agent
            self.selected_roles = self.role_selector.select_role(role_outputs, test_mode=test_mode,
                                                                 t_env=t_env).squeeze()
            # [bs * n_agents]

        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        roles_q = []
        # compute individual q-values for each agent
        for role_i in range(self.n_roles):
            role_q = self.roles[role_i](self.hidden_states, self.action_repr)  # [bs * n_agents, n_actions]
            roles_q.append(role_q)

        roles_q = th.stack(roles_q, dim=1)  # [bs*n_agents, n_roles, n_actions]

        # q value for each agent for each role
        agent_outs = th.gather(roles_q, 1, self.selected_roles.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.n_actions))

        # [bs * n_agents, n_roles , 1]

        # [bs * n_agents, n_roles, n_actions]
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        if role_outputs is not None:
            role_outputs = role_outputs.view(ep_batch.batch_size, self.n_agents, -1)

        return agent_outs, role_outputs

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
