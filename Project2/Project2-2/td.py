#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    best_action=np.argmax(Q[state])
    action=np.ones(nA)*epsilon/nA
    action[best_action]+= 1-epsilon

    return np.random.choice(np.arange(len(Q[state])), p=action)

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA= env.action_space.n
    for i in range(n_episodes):
        epsilon= 0.9*epsilon
        state_1 = env.reset()
        action_1=epsilon_greedy(Q, state_1, nA, epsilon)
        done= False
        while not done:
            state_2,reward,done, info,_=env.step(action_1)
            action_2=epsilon_greedy(Q, state_2, nA, epsilon)
            Q[state_1][action_1]=Q[state_1][action_1]+alpha*(reward+gamma*(Q[state_2][action_2])-Q[state_1][action_1])
            state_1=state_2
            action_1=action_2
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA= env.action_space.n
    for i in range(n_episodes):
        epsilon= 0.9*epsilon
        state_1 = env.reset()
        done= False
        while not done:
            action_1=epsilon_greedy(Q, state_1, nA, epsilon)
            state_2,reward,done, info,_=env.step(action_1)
            # action_2=epsilon_greedy(Q, state_2, nA, epsilon)
            q_star=np.argmax(Q[state_2])
            Q[state_1][action_1]=Q[state_1][action_1]+alpha*(reward+gamma*(Q[state_2][q_star])-Q[state_1][action_1])
            state_1=state_2
    return Q