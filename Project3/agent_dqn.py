#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque,namedtuple
from itertools import count
from typing import List
import os
import sys
import wandb

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


from agent import Agent
from dqn_model import DQN

wandb.init(project='N11_test2',entity='avarude')
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
#Params from the original paper
CONSTANT = 200000
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10000

#epsilon pararmeters for the decay
EPSILON = 1
EPSILON_END = 0.025
DECAY_EPSILON_AFTER = 3000
#updating the model params
TARGET_UPDATE_FREQUENCY = 5000
SAVE_MODEL_AFTER = 5000
# EPSILON_DECAY=0.5
#learning rate
LEARNING_RATE = 1.5e-4  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
reward_buffer = deque([0.0], maxlen=100)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        # self.EPSILON= args.EPSILON
        # self.EPSILON_END=args.EPSILON_END
        # self.EPSILON_DECAY=args.EPSILON_DECAY
        # self.DECAY_EPSILON_AFTER=args.DECAY_EPSILON_AFTER
        # self.TARGET_UPDATE_FREQUENCY=args.TARGET_UPDATE_FREQUENCY
        # self.BUFFER_SIZE=args.BUFFER_SIZE

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.env = env
        self.action_count = self.env.action_space.n
        in_channels = 4  # (R, G, B, Alpha)
        self.Q = DQN(in_channels, self.action_count).to(device)
        self.Q_cap = DQN(in_channels, self.action_count).to(device)
        self.Q_cap.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)
        
        # defining a buffer usind deque
        self.buffer=deque([], maxlen=BUFFER_SIZE)

        self.traning=0
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            test=torch.load('final_dqn_model.pth')
            self.Q.load_state_dict(test)
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        state=np.asarray(observation,dtype=np.float32)/255
        state=state.transpose(2,0,1)
        state=torch.from_numpy(state).unsqueeze(0)
        Q_new=self.Q(state)
        best_q_idx=torch.argmax(Q_new,dim=1)[0]
        action_idx=best_q_idx.detach().item()

        return action_idx
    
    def push(self,*args)->None:
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.buffer.append(Transition(*args))

        
        
    def replay_buffer(self,batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        samples=[]
        for i in range(batch_size):
            idx=random.randrange(len(self.buffer))
            samples.append(self.buffer[idx])
            del self.buffer[idx]
        return samples


    def train(self):
        """
        Implement your training algorithm here
        """
        episode=0
        exp_reward=0
        while np.mean(reward_buffer) < 50:
            
            print("Doin Episode", episode)
            t_stamp=0
            epi_reward=0
            #put counter ++ in the end of while loop
            curr_state=self.env.reset()

            while True:
                if episode > DECAY_EPSILON_AFTER:
                    epsilon = max(EPSILON_END,epsilon-(epsilon-EPSILON_END)/CONSTANT)
                else:
                    epsilon = EPSILON
                if random.random()>epsilon:
                    action=self.make_action(curr_state)
                else:
                    action =np.random.randint(0,4)
                next_state,reward,done,_,_=self.env.step(action)
                #changin the format of np array to tensors
                tensor_reward=torch.tensor([reward],device=device)
                tensor_action=torch.tensor([action],dtype=torch.int64,device=device)
                
                state=np.asarray(curr_state,dtype=np.float32)/255
                state=state.transpose(2, 0, 1)
                store_buffer_curr_state=torch.from_numpy(state).unsqueeze(0)
                
                state=np.asarray(next_state,dtype=np.float32)/255
                state=state.transpose(2, 0, 1)
                store_buffer_next_state=torch.from_numpy(state).unsqueeze(0)

                self.push(store_buffer_curr_state,tensor_action,tensor_reward,store_buffer_next_state)
                curr_state=next_state
                epi_reward+=reward
                if len(self.buffer)>=5000:
                    self.optimize()

                if done:
                    reward_buffer.append(epi_reward)
                    episode+=1
                    break
                t_stamp+=1

            if episode % 100 == 0:
                wandb.log({"reward":np.mean(reward_buffer),"episode":episode,"epsilon":epsilon,"timestamp":t_stamp})
            if episode % TARGET_UPDATE_FREQUENCY == 0:
                self.Q_cap.load_state_dict(self.Q.state_dict())
            if episode % SAVE_MODEL_AFTER == 0:
                # if exp_reward <= np.mean(reward_buffer):
                torch.save(self.Q.state_dict(), "final_dqn_model.pth")
                print("saving model at reward %f",np.mean(reward_buffer))
                exp_reward=np.mean(reward_buffer)
                wandb.log({"exp_reward":exp_reward})
                                

        torch.save(self.Q.state_dict(), "final_dqn_model.pth")
        print("Done Wooooo")    
    
    def optimize(self):
        print("optimizing")
        transitions = self.replay_buffer(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print(action_batch)
        # print("action batch shape")
        # print(action_batch.size())

        sav = self.Q(state_batch)
        # print(sav)
        # print("sav shape")
        # print(torch.arange(sav.size(0)))
        # state_action_values=sav.gather(1, action_batch) # according to pytorch tutorial it shoul work but had to hardcode the .gather function due to some errors
        state_action_values = sav[torch.arange(sav.size(0)), action_batch]
        # print(state_action_values)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.Q_cap(non_final_next_states).max(1)[0].detach()
        
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values = torch.reshape(
            expected_state_action_values.unsqueeze(1),
            (1, BATCH_SIZE)
        )[0]

        # Compute Huber loss
 
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        wandb.log({"loss":loss})

        # Optimize the model
 
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
 
        # return loss


