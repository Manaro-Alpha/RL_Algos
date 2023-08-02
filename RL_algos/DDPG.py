import gym
import os
import random
import time
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



class Q_net(nn.Module):
    def __init__(self,obs_dim,action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim+action_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    
    def forward(self,obs,action):
        return self.network(torch.cat([obs,action],0))

class Actor(nn.Module):
    def __init__(self,obs_dim,action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,action_dim)
        )

    def forward(self,obs):
        return self.network(obs)
    

def rollout(env:ObsAvExp1.Env,obs,obs_buffer,next_obs_buffer,action_buffer,reward_buffer,dones_buffer,global_step,buffer_size,actor:Actor,device):
    step = global_step % buffer_size
    with torch.no_grad():
        action = actor(torch.Tensor(obs).to(device))
        action += torch.normal(0,torch.Tensor([1,]))
        action = action.cpu().clip(env.min_vel,env.max_vel)
    action_buffer[step] = action
    next_obs,reward,done,truncated,info = env.step(action)
    obs_buffer[step] = torch.Tensor(obs)
    next_obs_buffer[step] = torch.Tensor(next_obs)
    reward_buffer[step] = torch.Tensor([reward])
    dones_buffer[step] = torch.Tensor([done])
    if done:
        obs,_ = env.reset()
    else:
        obs = next_obs
    return obs

def train(obs_buffer,next_obs_buffer,action_buffer,rewards_buffer,dones_buffer,buffer_size,batch_size,q_net,actor,actor_target,gamma,q_target,q_optim,actor_optim,tau):
    batch_i = random.sample(range(buffer_size),batch_size)
    obs = obs_buffer[batch_i]
    next_obs = next_obs_buffer[batch_i]
    action = action_buffer[batch_i]
    reward = rewards_buffer[batch_i]
    done = dones_buffer[batch_i]
    with torch.no_grad():
        a_tp1 = actor_target(next_obs)
        q_tp1 = q_target(next_obs,action)
        q_t_target = reward + (1-done)*gamma*q_tp1(next_obs,a_tp1)
    
    q_t = q_net(obs,action)
    q_loss = F.mse_loss(q_t,q_t_target)

    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()

    actor_loss = -q_net(obs,actor(obs)).mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    for param, target_param in zip(q_net.parameters(),q_target.parameters):
        target_param.data.copy_(tau*param.data + (1 - tau)*target_param.data)
    
    for param, target_param in zip(actor.parameters(),actor_target.parameters):
        target_param.data.copy_(tau*param.data + (1 - tau)*target_param.data)

def DDPG(env,actor,actor_target,q_net,q_target,actor_optim,q_optim,buffer_size,batch_size,total_timesteps,learning_starts,gamma,tau,device):
    obs_buffer = torch.zeros((buffer_size,env.obs_dim())) ## find a way to get obs and action shape from env
    next_obs_buffer = torch.zeros((buffer_size,env.obs_dim())).to(device)
    action_buffer = torch.zeros((buffer_size,env.action_dim())).to(device)
    reward_buffer = torch.zeros((buffer_size,)).to(device)
    dones_buffer = torch.zeros((buffer_size,)).to(device)

    obs,_ = env.reset()
    for global_step in range(total_timesteps):
        obs = rollout(env,obs,obs_buffer,next_obs_buffer,action_buffer,reward_buffer,dones_buffer,global_step,buffer_size,actor,device)

        if global_step > learning_starts:
            train(obs_buffer,next_obs_buffer,action_buffer,reward_buffer,dones_buffer,buffer_size,batch_size,q_net,actor,actor_target,gamma,q_target,q_optim,actor_optim,tau)


if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    run_name = run_name = f"ObsAvRL__1__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    ## add hyperparams to logger later

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor = Actor(env.obs_dim(),env.action_dim()).to(device)
    q_net = Q_net(env.obs_dim(),env.action_dim()).to(device)
    actor_target = Actor(env.obs_dim(),env.action_dim()).to(device)
    q_target = Q_net(env.obs_dim(),env.action_dim()).to(device)
    actor_target.load_state_dict(actor.state_dict())
    q_target.load_state_dict(q_net.state_dict())
    q_optim = optim.Adam(list(q_net.parameters()),lr = 3e-4)
    actor_optim = optim.Adam(list(actor.parameters()),lr = 3e-4)

    DDPG(
        env=env,
        actor=actor,
        actor_target=actor_target,
        q_net=q_net,
        q_target=q_target,
        actor_optim=actor_optim,
        q_optim=q_optim,
        buffer_size=int(25e3),
        batch_size=256,
        total_timesteps=int(1e6),
        learning_starts=25e3,
        gamma=0.99,
        tau=0.005,
        device=device,
    )