import torch.nn as nn
from torch.nn import ModuleList, Dropout, Linear, ReLU, BatchNorm1d


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, drop_hidden, layer_sizes):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = 2
        self.drop_input = 0.1
        self.drop_hidden = drop_hidden
        self.layer_sizes = layer_sizes
        self.batch_norm = False

        # Constructing network
        self.layers = []
        prev = self.input_shape[1]
        if self.drop_input > 0.:
            self.layers.append(Dropout(p=self.drop_input, inplace=False))

        for i in range(len(self.layer_sizes)):
            self.layers.append(Linear(prev, self.layer_sizes[i]))
            self.layers.append(ReLU())
            if self.drop_hidden[i] > 0.:
                self.layers.append(Dropout(p=self.drop_hidden[i], inplace=False))
            if self.batch_norm:
                self.layers.append(BatchNorm1d(self.layer_sizes[i]))
            prev = self.layer_sizes[i]

        self.layers = ModuleList(self.layers)
        self.output = Linear(self.layer_sizes[-1], self.output_shape)

    def forward(self, x):
        """
        x :  torch tensor
            The inputs to the model.
        """
        s = nn.Softmax(dim=1)
        for i, lay in enumerate(self.layers):
            x = lay(x)
        output = self.output(x)
        return s(output)