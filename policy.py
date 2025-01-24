import torch.nn as nn
import torch
import torch.nn.functional as F



class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_std, screen_size, num_click_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_click = nn.Linear(256, num_click_actions)
        self.fc_coord = nn.Linear(256, 2)
        self.action_std = action_std
        self.screen_width = screen_size
        self.screen_height = screen_size
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        click_logits = self.fc_click(x)
        click_probs = F.softmax(click_logits, dim=1)
        click_distribution = torch.distributions.Categorical(probs=click_probs)
        click_action = click_distribution.sample()
        log_prob_click = click_distribution.log_prob(click_action)
        coord_mean = self.fc_coord(x)
        coord_mean_x = torch.sigmoid(coord_mean[:, 0:1]) * self.screen_width
        coord_mean_y = torch.sigmoid(coord_mean[:, 1:2]) * self.screen_height
        coord_mean = torch.cat([coord_mean_x, coord_mean_y], dim=1)
        noise = torch.empty_like(coord_mean).uniform_(-self.action_std, self.action_std)
        coords = coord_mean + noise
        var = (self.action_std ** 2) / 3
        log_scale = torch.log(torch.tensor(self.action_std * (2 * np.sqrt(3))))
        log_prob_coord = - ((coords - coord_mean) ** 2) / (2 * var) - log_scale
        log_prob_coord = log_prob_coord.sum(dim=1)
        total_log_prob = log_prob_click + log_prob_coord
        return click_action, log_prob_click, coords, log_prob_coord, total_log_prob