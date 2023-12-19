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

def epsilon_greedy(action_logits, epsilon):
    # print(action_logits)
    sample = random.random()
    if sample < epsilon:
        action = random.randint(0, len(action_logits[0])-1)
    else:
        action = torch.argmax(action_logits).item()
    return action

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
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    # def action(self, obs): 
    #     Qs = self.forward(obs)
        
    #     return action  
    
    def get_Qs(self, obs):
        return self.forward(obs)
    
    def get_Q(self, obs, action):
        return self.forward(obs)[0][action]

class PolicyFC(nn.Module):
    def __init__(self, len_obs, len_act, len_hidden):
        super(PolicyFC, self).__init__()
        self.l1 = nn.Linear(len_obs, len_hidden)
        self.l2 = nn.Linear(len_hidden, len_act)    

    def forward(self, obs):
        obs = obs.reshape(1, -1)
        x = self.l1(obs)
        x = F.tanh(x)
        x = self.l2(x)
        return x
    
    def get_Qs(self, obs):
        return self.forward(obs)
    
    def get_Q(self, obs, action):
        return self.forward(obs)[0][action]

def downsample(obs): # obs is a 210x160 matrix
    return zoom(obs, ZOOM_FACTOR)

def preprocess(old_stack, new_frame):
    obs = downsample(new_frame)
    obs = torch.tensor(cut_to_square(obs)).unsqueeze(0)
    # print(obs.shape)
    # print(old_stack.shape)
    new_stack =torch.cat((old_stack[1:], obs), 0)
    # print(new_stack.shape)
    # new_stack = old_stack[1:].append(obs)

    return new_stack

def preprocess_FC(old_stack, new_frame):
    # print(new_frame, new_frame.shape)
    # print(old_stack, old_stack.shape)
    obs = torch.tensor(new_frame).unsqueeze(0)
    new_stack =torch.cat((old_stack[1:], obs), 0)
    return new_stack

def cut_to_square(obs):
    obs = obs[FRAME_84[0]:FRAME_84[1], :]
    return obs

def make_init_frame(obs, stack_length):
    obs = downsample(obs) #Init first input stack
    obs = cut_to_square(obs)
    obs = np.array([obs]*(stack_length))
    obs = torch.tensor(obs, dtype=torch.float32)
    return obs

def make_init_frame_FC(obs, stack_length):
    obs = np.array([obs]*(stack_length))
    obs = torch.tensor(obs, dtype=torch.float32)
    # print(obs)
    return obs


def playFC( env, policy, stack_length, epsilon):
    
    for i in range(10):
        obs, _ = env.reset()
        obs = make_init_frame_FC(obs, stack_length)
        terminal = False
        # print(obs)
        while not terminal:
            Qs = policy.get_Qs(obs)
            # print(Qs)
            action = epsilon_greedy(Qs, epsilon)
            new_frame, reward, terminal, _, info = env.step(action)
            obs = preprocess_FC(obs, new_frame)

def playCNN( env, policy, stack_length, epsilon):
    
    for i in range(10):
        obs, _ = env.reset()
        obs = make_init_frame(obs, stack_length)
        terminal = False
        # print(obs)
        while not terminal:
            Qs = policy.get_Qs(obs)
            # print(Qs)
            action = epsilon_greedy(Qs, epsilon)
            new_frame, reward, terminal, _, info = env.step(action)
            obs = preprocess(obs, new_frame)

if __name__ == '__main__':
    ID = 'ALE/SpaceInvaders-v5'
    ID_SIMPLE = 'LunarLander-v2'
    MODE = 0
    OBS_TYPE = 'grayscale'
    RENDER_MODE ='human'
    ZOOM_FACTOR = 0.5238
    FRAME_84 = (16, 100)
    FRAME_SQ_SIZE = 84   
    STACK_LENGTH = 4
    EPSILON = 0
    BATCH_SIZE = 16
    NUM_EPS = 10
    LR = 0.001
    GAMMA = 0.999
    EP_MAX_LEN = 1000
    MEM_SIZE = 1000
    MAX_TRAIN_STEPS = 1000

    ENV = gym.make( id = ID , mode = MODE, obs_type = OBS_TYPE, render_mode = RENDER_MODE )
    # ENV_SIMPLE = gym.make( id = ID_SIMPLE, render_mode = RENDER_MODE )

    # len_obs = ENV_SIMPLE.observation_space.shape[0]
    # len_act = ENV_SIMPLE.action_space.n
    # len_hidden = 48
    modelB = PolicyCNN(ENV.action_space.n, epsilon = EPSILON)

    # modelB = PolicyFC(len_obs=len_obs, len_act = len_act, len_hidden = len_hidden)
    modelB.load_state_dict(torch.load("dqn_saved_1"), strict=False)
    playCNN(ENV, modelB, STACK_LENGTH, EPSILON)
    # playFC(ENV_SIMPLE, modelB, STACK_LENGTH, EPSILON)

