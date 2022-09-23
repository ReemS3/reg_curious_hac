import torch
from torch import nn
from src.chac.utils import Base

class RepresentationNetwork(Base):
    """TODO: add the documentation here"""
    def __init__(self, env, layer, abs_range, out_dim):
        super(RepresentationNetwork, self).__init__()
        self.obs_dim = env.state_dim
        # Dimensions of action will depend on layer level
        self.out_dim = env.action_dim if layer == 0 else env.subgoal_dim
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
        # if layer > 2:
        #     for __ in range(layer - 2):
        #         obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if layer > 1:
            obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]
        
        self.obs_encoder = nn.Sequential(*obs_models)
        self.abs_range = abs_range
        self.representation_optim = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, obs):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        s = self.obs_encoder(obs)
        # this should not be implemented according to our project
        # return torch.tanh(s) * self.action_space_bounds + self.action_offset
        return s + self.action_offset

    def update(self, mu_loss):
        self.representation_optim.zero_grad()
        mu_loss.backward()
        flat_grads = torch.cat([param.flatten() for __, param in self.named_parameters()])
        self.representation_optim.step()
        return {
            'mu_loss': mu_loss.item(),
            'mu_grads': flat_grads.mean().item(),
            'mu_grads_std': flat_grads.std().item(),
        }