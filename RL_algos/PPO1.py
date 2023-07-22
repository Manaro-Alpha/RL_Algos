import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

global_step = 0

def make_envs(seed,idx,capture_video,run_name):
    def make_env():
        env = gym.make('LunarLander-v2')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return make_env

def layer_init(layer, std=np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent,self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1),std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n),std=0.01)
        )
    
    def get_value(self,obs):
        return self.critic(obs)

    def get_action_value(self,obs,action=None):
        logits = self.actor(obs)
        action_prob = Categorical(logits=logits) ##policy
        if action == None:
            action = action_prob.sample()
        return action, action_prob.log_prob(action), action_prob.entropy(), self.critic(obs)
        
def logger(writer,info,global_step):
    for item in info:
        if "episode" in item.keys():
            print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            break

def rollout_buffer(num_steps,num_envs,envs,gae,gamma,gae_lambda,agent,device,writer):
    global global_step
    global_step += num_envs
    ### removed .to(device). Add later
    obs = torch.zeros(((num_steps, num_envs) + envs.single_observation_space.shape)).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)

    for step in range(0,num_steps):
        obs[step] = next_obs
        dones[step] = next_done
        with torch.no_grad():
            action,logprob,_,value = agent.get_action_value(obs=next_obs)
        actions[step] = action
        logprobs[step] = logprob
        values[step] = value.flatten()

        next_obs,reward,done,info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).view(-1)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor(done).to(device)

        # for item in info:
        #     if 'episode' in item.keys():
        #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
        #         break

    with torch.no_grad():
        next_value = agent.get_value(obs=next_obs).reshape(1,-1)
        if gae:
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps-1:
                    nextnonterminal = 1 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + nextnonterminal*gamma*nextvalues - values[t]
                advantages[t] = last_gae_lam = delta + nextnonterminal * last_gae_lam * gamma * gae_lambda
            returns = advantages + values
        
        else:
            returns = torch.zeros_like(rewards)
            for t in reversed(range(num_steps)):
                if t == num_steps -1:
                    nextnonterminal = 1 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1 - dones[t+1]
                    nextvalues = values[t+1]
                returns[t] = rewards[t] + nextnonterminal*gamma*nextvalues
            advantages = returns - values

    return obs,logprobs,actions,advantages,returns,values

def train(
        obs,logprobs,actions,advantages,returns,values,envs,optimizer,batch_size,minibatch_size,clip_coef,update_epochs,agent,
        norm_adv=True,clip_vloss=True,ent_coef=0.01,vf_coef=0.5,max_grad_norm=0.5,target_kl=None):
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1,)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1,)
    b_returns = returns.reshape(-1,)
    b_values = values.reshape(-1,)

    #generating random indices for minibatch training
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0,batch_size,minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            _,new_logprob,entropy,new_value = agent.get_action_value(obs=b_obs[mb_inds],action=b_actions.long()[mb_inds])
            log_ratio = new_logprob - b_logprobs[mb_inds]
            ratio = log_ratio.exp()

            with torch.no_grad():
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
            
            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Actor(Policy) loss
            pg_loss1 = -mb_advantages*ratio
            pg_loss2 = -mb_advantages*torch.clamp(ratio,1-clip_coef,1+clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Critic(Value) loss
            new_value = new_value.view(-1)
            if clip_vloss:
                v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds],-clip_coef,clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
            
            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + vf_coef*v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break
    return pg_loss,v_loss,entropy_loss,old_approx_kl,approx_kl,clipfracs

def ppo(gym_id,num_envs,lr,total_timesteps,num_steps,batch_size,minibatch_size,gae,gamma,gae_lambda,
        clip_coef,update_epochs,seed=1,torch_deterministic=True):
    exp_name = os.path.basename(__file__).rstrip(".py")
    run_name = f"{'LunarLander-v3'}__{exp_name}__{seed}__{int(time.time())}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{run_name}")
    envs = gym.vector.SyncVectorEnv(make_envs(seed+i,i,True,run_name) for i in range(num_envs))

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    start_time = time.time()
    num_updates = total_timesteps//batch_size
    for update in range(1, num_updates+1):
        obs,logprobs,actions,advantages,returns,values = rollout_buffer(num_steps,num_envs,envs,gae,gamma,gae_lambda,agent,device,writer)
        b_returns = returns.reshape(-1,)
        b_values = values.reshape(-1,)
        pg_loss,v_loss,entropy_loss,old_approx_kl,approx_kl,clipfracs = train(obs,logprobs,actions,advantages,returns,values,envs,
                                                                            optimizer,batch_size,minibatch_size,clip_coef,update_epochs,agent)
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    envs.close()
    writer.close()

if __name__ == '__main__':
    ppo(gym_id='CartPole-v1',
        num_envs=3,
        lr=2.5e-4,
        total_timesteps=25000,
        num_steps=128,
        batch_size=3*128,
        minibatch_size=4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=4)