import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

class Q_Networks(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )
    
    def forward(self, obs):
        return self.network(obs)
    
def act(envs,q_net,obs,epsillon,device):
    if random.random() < epsillon:
        action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        q_values = q_net.forward(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values,dim=1).cpu().numpy()
    return action

def initialise(envs,buffer_size,device):
    rb = ReplayBuffer(
        buffer_size,envs.single_observation_space,envs.single_action_space,device,handle_timeout_termination=False
    )

    obs,_ = envs.reset()
    return obs

def rollout(envs,q_net,obs,epsillon,rb,device):
    action = act(envs,q_net,obs,epsillon,device)
    next_obs,reward,done,truncated,info = envs.step(action)
    real_next_obs = next_obs.copy()
    for idx,d in enumerate(truncated):
        if d:
            real_next_obs[idx] = info["final_observation"][idx]
    rb.add(obs,next_obs,action,reward,done,info)
    obs = next_obs
     ## adjust the last episode according to gym api later
    ## obs is given as an input. initialise and update obs in the main loop

def train(rb,q_net,target_net,optimizer,gamma,batch_size):
    data = rb.sample(batch_size)
    with torch.no_grad():
        Q_t,_ = target_net(data.next_observations).max(dim=1)
        target = data.rewards.flatten() + gamma*Q_t*(1-data.dones.flatten())
    
    Q_value = q_net(data.observations).gather(1,data.actions).squeeze()
    loss = F.mse_loss(Q_value,target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss,Q_value

def update(q_net,target_net):
    for target_network_param, q_network_param in zip(target_net.parameters(), q_net.parameters()):
        target_network_param.data.copy_(q_network_param.data)

def DQN(
        envs,q_net,target_net,buffer_size,epsillon,gamma,total_timesteps,batch_size,start_train,update_freq,optimizer,device,writer,save_model,run_name,exp_name
):
    rb = ReplayBuffer(
        buffer_size,envs.single_observation_space,envs.single_action_space,device,handle_timeout_termination=False
    )
    obs,_ = envs.reset(seed=seed)
    start_time = time.time()
    for global_step in range(total_timesteps):
        rollout(envs,q_net,obs,epsillon,rb,device)
        if global_step > start_train:
            loss,Q_value = train(rb,q_net,target_net,optimizer,gamma,batch_size)
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", Q_value.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if global_step % update_freq == 0:
            update(q_net,target_net)
        
    if save_model == True:
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        torch.save(q_net.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

if __name__ == '__main__':
    env_id = 'CartPole-v1'

    seed = 1
    exp_name = os.path.basename(__file__).rstrip(".py")
    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    learning_rate = 2.5e-4

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, False, run_name) for i in range(1)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_net = Q_Networks(envs).to(device)
    optimizer = optim.Adam(q_net.parameters())
    target_net = Q_Networks(envs).to(device)
    target_net.load_state_dict(q_net.state_dict())

    DQN(
        envs = envs,
        q_net=q_net,
        target_net=target_net,
        buffer_size=10000,
        epsillon=0.05,
        gamma=0.99,
        total_timesteps=500000,
        batch_size=128,
        start_train=10000,
        update_freq=500,
        optimizer=optimizer,
        device=device,
        writer=writer,
        save_model=True,
        run_name=run_name,
        exp_name=exp_name
    )


