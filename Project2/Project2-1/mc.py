#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    if(score>=20):
        action=0
    else:
        action=1

    # action

    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    for i in range(n_episodes):
        obs=env.reset()
        episode=[]
        while True:
            # print(c)
            action=policy(obs)
            new_state, reward,done,desc,_=env.step(action)
            episode.append((obs,action,reward))
            if done:
                break
            obs=new_state
        rev_episode=reversed(episode)
        G_states=[]
        G=0
        for (obs,action,reward) in rev_episode:
            G= gamma*G+reward
            G_states.append(G)
        G_states.reverse()
        visited_states=[]
        for index,(obs,action,reward) in enumerate(episode):
            if obs in visited_states:
                continue
            visited_states.append(obs)
            returns_sum[obs]+=G_states[index]
            returns_count[obs] += 1
            V[obs]=returns_sum[obs]/returns_count[obs]
            

    return V

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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    best_action=np.argmax(Q[state])
    action=np.ones(nA)*epsilon/nA
    action[best_action]+= 1-epsilon
    return np.random.choice(np.arange(len(Q[state])), p=action)

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    nA= env.action_space.n
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    while epsilon > 0:
        for i in range(n_episodes):
            obs=env.reset()
            episode=[]
            while True:
                # print(c)
                action=epsilon_greedy(Q, obs, nA, epsilon)
                new_state, reward,done,desc,_=env.step(action)
                episode.append((obs,action,reward))
                if done:
                    break
                obs=new_state
            rev_episode=reversed(episode)
            G_rewards=[]
            G=0
            for (obs,action,reward) in rev_episode:
                G= gamma*G+reward
                G_rewards.append(G)
            G_rewards.reverse()
            visited_states=[]
            visited_actions=[]
            for index,(obs,action,reward) in enumerate(episode):
                if obs in visited_states and action in visited_actions:
                    continue
                visited_states.append(obs)
                visited_actions.append(action)
                returns_sum[(obs,action)]+=G_rewards[index]
                returns_count[(obs,action)] += 1
                Q[obs][action]=returns_sum[(obs,action)]/returns_count[(obs,action)]
            epsilon=epsilon-0.1/n_episodes

    return Q
