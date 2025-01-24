import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from bert import BertEncoder
from env import DesktopEnv
from resnet import ResNet18FeatureExtractor
from value_net import ValueNetwork
from policy import PolicyNetwork

LEARNING_RATE_POLICY = 1e-4
LEARNING_RATE_VALUE = 1e-4
NUM_EPISODES = 500
GAMMA = 0.99
ACTION_STD = 100.0
SCREEN_SIZE = 1000.0
NUM_CLICK_ACTIONS = 4
MAX_STEPS = 1000
TARGET_REGION = (100, 100)
REGION_SIZE = 10.0


def process_state(state, bert_encoder, resnet_extractor):
    image, command = state
    img_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
    cmd_tensor = bert_encoder([command])
    img_feat = resnet_extractor(img_tensor)
    return torch.cat((img_feat, cmd_tensor), dim=1)


def compute_returns_advantages(transitions, gamma, value_net, bert_encoder, resnet_extractor):
    rewards = [t[2] for t in transitions]
    states = [t[0] for t in transitions]
    next_states = [t[3] for t in transitions]
    values, next_values = [], []
    
    with torch.no_grad():
        values = [value_net(process_state(s, bert_encoder, resnet_extractor)).item() for s in states]
        next_values = [value_net(process_state(ns, bert_encoder, resnet_extractor)).item() for ns in next_states]

    returns, advantages = [], []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] - values[i]
        gae = delta + gamma * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return returns, advantages


def update_policy_value(policy_net, value_net, bert_encoder, resnet_extractor, transitions, returns, advantages, optimizer_policy, optimizer_value):
    policy_loss, value_loss = 0.0, 0.0
    
    for i, (state, log_prob, _, _) in enumerate(transitions):
        G = torch.tensor(returns[i], dtype=torch.float32)
        A = torch.tensor(advantages[i], dtype=torch.float32)
        policy_loss += -log_prob * A
        cf = process_state(state, bert_encoder, resnet_extractor)
        value_est = value_net(cf)
        value_loss += F.mse_loss(value_est, G.unsqueeze(0))

    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    return policy_loss.item(), value_loss.item()


def train_agent(gamma=GAMMA):
    bert_encoder = BertEncoder()
    resnet_extractor = ResNet18FeatureExtractor()
    input_dim = bert_encoder.hidden_dim + 512
    policy_net = PolicyNetwork(input_dim=input_dim, action_std=ACTION_STD, screen_size=SCREEN_SIZE, num_click_actions=NUM_CLICK_ACTIONS)
    value_net = ValueNetwork(input_dim)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE_POLICY)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE_VALUE)

    env = DesktopEnv(target_region=TARGET_REGION, max_steps=MAX_STEPS, region_size=REGION_SIZE)
    
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        transitions, total_reward, steps = [], 0.0, 0

        while True:
            combined_feat = process_state(state, bert_encoder, resnet_extractor)
            click_action, log_prob_click, coords, log_prob_coord, total_log_prob = policy_net(combined_feat)
            next_state, reward, done, _ = env.step(click_action, coords)
            transitions.append((state, total_log_prob, reward, next_state))
            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        returns, advantages = compute_returns_advantages(transitions, gamma, value_net, bert_encoder, resnet_extractor)
        p_loss, v_loss = update_policy_value(policy_net, value_net, bert_encoder, resnet_extractor, transitions, returns, advantages, optimizer_policy, optimizer_value)

        policy_net.action_std = max(1.0, policy_net.action_std * 0.99)
        print(f"Episode {episode}, Steps {steps}, Reward: {total_reward:.3f}, PolicyLoss: {p_loss:.4f}, ValueLoss: {v_loss:.4f}")

        if steps < 20:
            torch.save(policy_net.state_dict(), "policy_net.pth")
            torch.save(value_net.state_dict(), "value_net.pth")
            print("Training finished, model saved.")
            break


if __name__ == "__main__":
    train_agent()
