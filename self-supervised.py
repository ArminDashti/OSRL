import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from bert import BertEncoder
from resnet import ResNet18FeatureExtractor

# Load and preprocess data
def load_human_interactions(filepath):
    with open(filepath, "rb") as f:
        interactions = pickle.load(f)
    return interactions

def preprocess_interactions(interactions):
    states, goals, commands, actions, click_actions = [], [], [], [], []

    for state, action, goal_state, click_action in interactions:
        states.append(torch.FloatTensor(state).permute(2, 0, 1))  # Convert image to tensor
        goals.append(torch.FloatTensor(goal_state).permute(2, 0, 1))
        commands.append(action)  # Encoded textual commands
        actions.append(torch.FloatTensor(action))  # Mouse actions
        click_actions.append(click_action)  # Click actions

    return states, goals, commands, actions, click_actions

class InteractionDataset(Dataset):
    def __init__(self, states, goals, commands, actions, click_actions):
        self.states = states
        self.goals = goals
        self.commands = commands
        self.actions = actions
        self.click_actions = click_actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.goals[idx], self.commands[idx], self.actions[idx], self.click_actions[idx]

# Neural network definition
class StateGoalActionNetwork(nn.Module):
    def __init__(self, state_dim=512, goal_dim=512, command_dim=768, hidden_dim=256, action_dim=2, click_dim=4):
        super(StateGoalActionNetwork, self).__init__()
        self.state_encoder = ResNet18FeatureExtractor(output_dim=state_dim)
        self.goal_encoder = ResNet18FeatureExtractor(output_dim=goal_dim)
        self.command_encoder = BertEncoder()
        
        self.fc1 = nn.Linear(state_dim + goal_dim + command_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.click_head = nn.Linear(hidden_dim, click_dim)

    def forward(self, state_image, goal_image, command_text):
        state_features = self.state_encoder(state_image)
        goal_features = self.goal_encoder(goal_image)
        command_features = self.command_encoder([command_text])
        
        combined_features = torch.cat((state_features, goal_features, command_features), dim=1)
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        
        actions = self.action_head(x)
        click_logits = self.click_head(x)
        click_probs = F.softmax(click_logits, dim=1)
        
        return actions, click_probs

# Main script
if __name__ == "__main__":
    # Filepath to human interactions
    filepath = "c:/users/armin/envenv/human_interactions.pkl"
    
    # Load and preprocess data
    interactions = load_human_interactions(filepath)
    states, goals, commands, actions, click_actions = preprocess_interactions(interactions)

    # Create dataset and dataloader
    dataset = InteractionDataset(states, goals, commands, actions, click_actions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss functions
    model = StateGoalActionNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_action = nn.MSELoss()
    criterion_click = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(10):  # Adjust epochs as needed
        total_loss = 0
        for state, goal, command, target_action, target_click in dataloader:
            optimizer.zero_grad()

            pred_action, pred_click = model(state, goal, command)
            loss_action = criterion_action(pred_action, target_action)
            loss_click = criterion_click(pred_click, target_click)
            loss = loss_action + loss_click

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
