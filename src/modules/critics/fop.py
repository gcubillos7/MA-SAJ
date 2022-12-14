import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FOPCritic(nn.Module):
    def __init__(self, scheme, args, n_actions = None):
        super(FOPCritic, self).__init__()

        self.args = args
        if n_actions is None:
            self.n_actions = args.n_actions
        else:
            self.n_actions = n_actions
            
        self.n_agents = args.n_agents

        # obs + n_agents
        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q  # bs, max_t, n_agents, n_actions

    def _build_inputs(self, batch, bs, max_t):

        inputs = [batch['obs'][:],
                  th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)]
        # state, obs, action

        # inputs[0] --> [bs, max_t, n_agents, obs]
        # inputs[1] --> [bs, max_t, n_agents, n_agents]

        # one hot encoded position of agent + state
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs  # [bs, max_t, n_agents, n_agents + n_obs]

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["obs"]["vshape"]
        input_shape += self.n_agents
        return input_shape  # [n_agents + n_obs]

# rnn --> dot --> q
