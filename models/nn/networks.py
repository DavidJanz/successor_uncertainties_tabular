from collections import OrderedDict

from torch import nn

from models.nn.utils import CloneMixin

activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}


class LinearActive(nn.Sequential):
    def __init__(self, layer_sizes, act=None):
        layers = OrderedDict()

        if act == None or act == 'none':
            activation = nn.ReLU
        elif act in activations:
            activation = activations[act]
        else:
            activation = act

        input_size = layer_sizes[0]
        for i, layer_size in enumerate(layer_sizes[1:]):
            layers[f'lin{i}'] = nn.Linear(input_size, layer_size)
            layers[f'act{i}'] = activation()
            input_size = layer_size

        super().__init__(layers)

    def init_xavier_(self):
        for name, tensor in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)


class FeatureLinear(nn.Module, CloneMixin):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()

        self.__args__ = (input_size, output_size, hidden_sizes)

        if hidden_sizes:
            self._features = LinearActive([input_size] + hidden_sizes)
            self.linear = nn.Linear(hidden_sizes[-1], output_size)
        else:
            self._features = lambda x: x
            self.linear = nn.Linear(input_size, output_size)

        self.init_xavier_()

    def init_xavier_(self):
        for name, tensor in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)

    def forward(self, x):
        features = self._features(x)
        return self.linear(features)

    def featurise(self, x):
        return self._features(x)
