import gymnasium as gym
import random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from collections import deque
import time




def downsample(obs): # obs is a 210x160 matrix
    return zoom(obs, ZOOM_FACTOR)

def preprocess(old_stack, new_frame):
    obs = downsample(new_frame)
    obs = torch.tensor(cut_to_square(obs)).unsqueeze(0)
    new_stack =torch.cat((old_stack[1:], obs), 0)
    
    return new_stack

def cut_to_square(obs):
    obs = obs[FRAME_84[0]:FRAME_84[1], :]
    return obs

def epsilon_greedy( action_logits, epsilon):
    sample = random.random()
    if sample < epsilon:
        action = random.randint(1, len(action_logits[0])-1)
    else:
        # print(action_logits[0][1:])
        action = torch.argmax(action_logits)
        # action = torch.argmax(action_logits[0][1:])
        # print(action)
    return action

class PolicyFC(nn.Module):
    def __init__(self, len_obs, len_act, len_hidden):
        super(PolicyFC, self).__init__()
        self.l1 = nn.Linear(len_obs, len_hidden)
        self.l2 = nn.Linear(len_hidden, len_act)    

    def forward(self, obs):
        x = self.l1(obs)
        x = F.tanh(x)
        x = self.l2(x)
        return x
    
    def get_Qs(self, obs):
        return self.forward(obs)
    
    def get_Q(self, obs, action):
        return self.forward(obs)[0][action]
    
    # def action(self, obs): 
    #     obs = obs.float().unsqueeze(0)
    #     logits = self.forward(obs)
    #     act_dist = Categorical(logits=logits)
    #     action_selected = act_dist.sample()
    #     return action_selected.item(), act_dist.log_prob(action_selected)

class PolicyCNN(nn.Module):
    def __init__(self, num_actions, epsilon):
        super(PolicyCNN, self).__init__()
        self.epsilon = epsilon

        self.conv1 = nn.Conv2d(4, 16, kernel_size= 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32,kernel_size= 4, stride = 2)
        self.fc1 = nn.Linear(32 * 9*9, 256)
        self.fc2 = nn.Linear(256,num_actions)

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_Qs(self, obs):
        # return self.forward(obs.to(device))
        return self.forward(obs)
    
    def get_Q(self, obs, action):
        # return self.forward(obs.to(device))[0][action]
        return self.forward(obs)[0][action]

class ReplayMemory():
    def __init__(self, mem_size):
        self.storage = deque(maxlen = mem_size)
    
    def store(self, obs):
        self.storage.append(obs)
    
    def elems_in_mem(self):
        return len(self.storage)
    
    def get_samples(self, num_samples):
        random_idxs = random.choices(range(self.elems_in_mem()), k = num_samples)
        
        transitions = []
        for idx in random_idxs:
            transitions.append(self.storage[idx])
        return transitions
    
def loss_fn(y, transitions , policy):
    sum = 0
    obs_idx = 0
    action_idx = 1
    for i in range(len(y)):
        sum += (y[i] - policy.get_Q(transitions[i][obs_idx],transitions[i][action_idx]))**2
    return sum

def reward_policy(reward, new_lives, prev_lives):
    if prev_lives > new_lives:
        reward = -100
    else:
        if reward > 0:
            reward = reward
        else:
           reward = 0
    return reward

def make_init_frame(obs, stack_length):
    obs = downsample(obs) #Init first input stack
    obs = cut_to_square(obs)
    obs = np.array([obs]*(stack_length))
    obs = torch.tensor(obs, dtype=torch.float32)
    return obs

def train( env, policy, batch_size, stack_length, max_steps, lr, gamma, ep_max_len,mem_size, epsilon):
    # plt.ion()

    fig, (ax_reward, ax_ep_length, ax_rew_ep) = plt.subplots(3, 1)
    x_reward = []
    x_ep_length = []
    x_rew_ep = []

    y_reward = [0]*len(x_reward)
    y_ep_length = [0]*len(x_ep_length)
    y_rew_ep = [0]*len(x_rew_ep)

    y_reward_max = 0
    y_ep_length_max = 0
    y_rew_ep_max = 0

    line_reward, = ax_reward.plot(x_reward, y_reward, label = 'Reward')
    line_ep_length, = ax_ep_length.plot(x_ep_length, y_ep_length, label = 'Episode Length')
    line_rew_ep, = ax_rew_ep.plot(x_rew_ep, y_rew_ep, label = 'Reward per Episode')


    # scores_deque = deque(maxlen = 100)
    scores_deque = []
    loss_deque = []
    ep_lengths = []
    num_steps = 0
    optimiser = torch.optim.Adam(policy.parameters(), lr)
    
    memory = ReplayMemory(mem_size).to(device)
    epsilon_end = epsilon
    epsilon_start = 1
    decay_factor = (epsilon_end/epsilon_start)**(1/max_steps)

    eps = epsilon_start
    num_episodes = 0
    while num_steps<max_steps:
        obs, _ = env.reset()
        obs = make_init_frame(obs, stack_length)
        terminal = False
        prev_lives = 3
        new_ep_flag = num_steps
        time_s = time.time()

        while not terminal:
            num_steps += 1
            # print(decay_factor)
            eps = eps * decay_factor
            # print(eps)

            Qs = policy.get_Qs(obs)
            action = epsilon_greedy(Qs, eps)

            new_frame, reward, terminal, _, info = env.step(action)
            reward = reward_policy(reward, info['lives'], prev_lives)
            prev_lives = info['lives']

            new_obs = preprocess(obs, new_frame)
            new_obs
            memory.store([obs, action, reward, terminal, new_obs])
            scores_deque.append(reward)
 
            obs = new_obs
            if memory.elems_in_mem()<batch_size:
                recalled_trans = memory.get_samples(memory.elems_in_mem())
            else:
                recalled_trans = memory.get_samples(batch_size)
            y = [0] * len(recalled_trans)
            reward_idx = 2
            terminal_idx = 3
            new_obs_idx = 4

            for i in range(len(recalled_trans)):
                if recalled_trans[i][terminal_idx] == True:
                    y[i] = recalled_trans[i][reward_idx]
                else:
                    with torch.no_grad():
                        Qs = policy.get_Qs(recalled_trans[i][new_obs_idx])
                    max_Q = torch.argmax(Qs)
                    y[i] = recalled_trans[i][reward_idx] + gamma * Qs[0][max_Q]
            # print(y)
            loss = loss_fn(y, recalled_trans, policy = policy )
            loss_deque.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            # if num_steps%100 == 0:
            #     y_reward.append(np.mean(scores_deque))
            #     x_reward.append(num_steps)
            #     ax_reward.set_xlim(0,x_reward[-1])
            #     if y_reward_max < y_reward[-1]:
            #         y_reward_max = y_reward[-1]
            #         ax_reward.set_ylim(0, y_reward_max)
            #     line_reward.set_ydata(y_reward)
            #     line_reward.set_xdata(x_reward)

        
                # plt.draw()
                # plt.pause(1)

                # print('Steps {}\tAverage Score: {:.2f}'.format(num_steps, np.mean(scores_deque)))
        num_episodes += 1
        print(f'Episode {num_episodes} {num_steps-new_ep_flag}\t T-Steps {num_steps}\tAverage Score: {np.sum(scores_deque):9.4f}\t  Epsilon: {eps:9.4f}\t Loss: {np.mean(loss_deque):9.4F}')        # ep_lengths.append(num_steps-ep_len)
        print(f'Time: {(time.time()-time_s)/(num_steps-new_ep_flag):9.4f}')
        # y_reward.append(np.mean(scores_deque))
        # x_reward.append(num_episodes)
        # ax_reward.set_xlim(0,x_reward[-1])
        # if y_reward_max < y_reward[-1]:
        #     y_reward_max = y_reward[-1]
        #     ax_reward.set_ylim(0, y_reward_max)
        # line_reward.set_ydata(y_reward)
        # line_reward.set_xdata(x_reward)
        # ax_reward.legend()
        
        # y_ep_length.append(num_steps-new_ep_flag)
        # x_ep_length.append(num_episodes)
        # ax_ep_length.set_xlim(0,num_episodes)
        # if y_ep_length_max < y_ep_length[-1]:
        #     y_ep_length_max = y_ep_length[-1]
        #     ax_ep_length.set_ylim(0, y_ep_length_max)
        # line_ep_length.set_ydata(y_ep_length)
        # line_ep_length.set_xdata(x_ep_length)
        # ax_ep_length.legend()

        # y_rew_ep.append(y_reward[-1]/y_ep_length[-1])
        # x_rew_ep.append(num_episodes)
        # ax_rew_ep.set_xlim(0,x_rew_ep[-1])
        # if y_rew_ep_max < y_rew_ep[-1]:
        #     y_rew_ep_max = y_rew_ep[-1]
        #     ax_rew_ep.set_ylim(0, y_rew_ep_max)
        # line_rew_ep.set_ydata(y_rew_ep)
        # line_rew_ep.set_xdata(x_rew_ep)
        # ax_rew_ep.legend()
        # if (num_episodes%10 == 0):
        #     plt.draw()
        #     plt.pause(1)
        scores_deque = []
        loss_deque = []
        if num_steps%1000000 == 0:
            torch.save(policy.state_dict(), f'dqn_saved_{num_steps}')


    # plt.ioff()
    # plt.show()
        # print('Episode Len {}'.format(num_steps-ep_len))



if __name__ == '__main__':
    
    print(torch.backends.mps.is_available())
    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")

    ID = 'ALE/SpaceInvaders-v5'
    # ID = 'LunarLander-v2'
    MODE = 0
    OBS_TYPE = 'grayscale'
    RENDER_MODE ='human'
    ZOOM_FACTOR = 0.5238
    FRAME_84 = (16, 100)
    FRAME_SQ_SIZE = 84   
    STACK_LENGTH = 4
    EPSILON = 0.05
    BATCH_SIZE = 32
    NUM_EPS = 10
    LR = 0.0005
    GAMMA = 0.999
    EP_MAX_LEN = 1000
    MEM_SIZE = 1000000
    MAX_TRAIN_STEPS = 10000000

    ENV = gym.make( id = ID , mode = MODE, obs_type = OBS_TYPE )
    
    POLICY_CNN = PolicyCNN(ENV.action_space.n, epsilon = EPSILON).to(device)
    # POLICY_CNN = PolicyCNN(ENV.action_space.n, epsilon = EPSILON)
    # POLICY_FC = PolicyFC(ENV.action_space.n, epsilon = EPSILON)
    
    # POLICY = torch.load('dqn_saved')
    train( env = ENV,
           policy= POLICY_CNN, 
           batch_size = BATCH_SIZE, 
           stack_length= STACK_LENGTH, 
           max_steps = MAX_TRAIN_STEPS, 
           lr = LR, 
           gamma = GAMMA, 
           ep_max_len = EP_MAX_LEN, 
           epsilon = EPSILON, 
           mem_size= MEM_SIZE
    )
    # torch.save(POLICY, "dqn_saved")
    torch.save(POLICY_CNN.state_dict(), "dqn_saved_final")
    