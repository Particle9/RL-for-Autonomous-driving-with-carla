import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from actor import Actor
from critic import Critic

class SAC(nn.Module):
    def __init__(self, image_shape=(480, 640), pointnet_weights=None, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005, target_entropy=-3):
        super(SAC, self).__init__()
        
        # Actor network
        self.actor = Actor(image_shape=image_shape, pointnet_weights=pointnet_weights)

        # Two critic networks for Q-value estimation
        self.critic1 = Critic(image_shape=image_shape, pointnet_weights=pointnet_weights)
        self.critic2 = Critic(image_shape=image_shape, pointnet_weights=pointnet_weights)

        # Target critic networks (for soft updates)
        self.critic1_target = Critic(image_shape=image_shape, pointnet_weights=pointnet_weights)
        self.critic2_target = Critic(image_shape=image_shape, pointnet_weights=pointnet_weights)
        self.update_target_networks(soft_update=False)

        # Optimizers for actor, critic, and temperature
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Entropy coefficient (alpha)
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update factor for target networks
        self.target_entropy = target_entropy  # Target entropy for automatic temperature tuning

    def update_target_networks(self, soft_update=True):
        tau = self.tau if soft_update else 1.0  # If not soft updating, perform hard update
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, image_input, point_input, eval_mode=False):
        with torch.no_grad():
            action = self.actor(image_input, point_input)
        return action if not eval_mode else action.clamp(-1.0, 1.0)  # Ensure action is in valid range during evaluation

    def compute_critic_loss(self, image_input, point_input, action, reward, next_image_input, next_point_input, done):
        with torch.no_grad():
            # Target actions and Q-values for the next state
            next_action = self.actor(next_image_input, next_point_input)
            target_q1 = self.critic1_target(next_image_input, next_point_input, next_action)
            target_q2 = self.critic2_target(next_image_input, next_point_input, next_action)
            target_q_value = reward + (1.0 - done) * self.gamma * torch.min(target_q1, target_q2)

        # Current Q-values
        current_q1 = self.critic1(image_input, point_input, action)
        current_q2 = self.critic2(image_input, point_input, action)

        # Critic loss as the mean squared error between current and target Q-values
        critic1_loss = F.mse_loss(current_q1, target_q_value)
        critic2_loss = F.mse_loss(current_q2, target_q_value)
        
        return critic1_loss, critic2_loss

    def compute_actor_and_alpha_loss(self, image_input, point_input):
        # Get actions from the actor
        action = self.actor(image_input, point_input)

        # Compute Q-values from both critics for the current state and action
        q1_value = self.critic1(image_input, point_input, action)
        q2_value = self.critic2(image_input, point_input, action)
        min_q_value = torch.min(q1_value, q2_value)

        # Compute the actor loss using the entropy-regularized policy loss
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * torch.log(action + 1e-10) - min_q_value).mean()

        # Temperature (alpha) loss for entropy regularization
        alpha_loss = -(self.log_alpha * (torch.log(action + 1e-10) + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss

    def update(self, image_input, point_input, action, reward, next_image_input, next_point_input, done):
        # Update critics
        critic1_loss, critic2_loss = self.compute_critic_loss(image_input, point_input, action, reward, next_image_input, next_point_input, done)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update actor and entropy temperature (alpha)
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(image_input, point_input)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        self.update_target_networks(soft_update=True)
