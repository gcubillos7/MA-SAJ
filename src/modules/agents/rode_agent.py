import torch.nn as nn
import torch.nn.functional as F


class RODEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RODEAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        return h
    


