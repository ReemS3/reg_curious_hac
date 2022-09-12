import torch
from torch import nn

class RepresentationNetwork(nn.Module):
    """TODO: add the documentation here"""
    def __init__(self, env, layer, abs_range, out_dim):
        super(RepresentationNetwork, self).__init__()
        self.obs_dim = env.state_dim
        self.out_dim = out_dim
        self.mid_dim = 100

        # Determine range of actor network outputs.
        self.action_space_bounds = torch.FloatTensor(env.action_bounds
                if layer == 0 else env.subgoal_bounds_symmetric)
        self.action_offset = torch.FloatTensor(env.action_offset
                if layer == 0 else env.subgoal_bounds_offset)

        if layer == 1:
            obs_models = [nn.Linear(self.obs_dim, self.out_dim)]
        else:
            obs_models = [nn.Linear(self.obs_dim, self.mid_dim)]
        if layer > 2:
            for __ in range(layer - 2):
                obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if layer > 1:
            obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.obs_encoder = nn.Sequential(*obs_models)
        self.abs_range = abs_range
        self.representation_optim = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, obs):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        s = self.obs_encoder(obs)
        return s * self.action_space_bounds + self.action_offset