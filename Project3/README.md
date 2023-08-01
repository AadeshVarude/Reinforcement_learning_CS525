# Inidividual Project 3
# Deep Q-learning Network(DQN)
  
  **Leaderboard for Breakout-DQN** 
  **Update Date: **
  
  | Top | Date | Name | Score | Model |
  | :---: | :---:| :---: | :---: | :---: |
  | 1  | 10/24/2022 | Palawat Busaranuvong | 317 | Prioritized DQN |
  | 2  | 11/15/2022 | Amey Deshpande | 166.8 | ... |
  | 2  | 11/15/2022 | Rane, Bhushan | 166.8 | ... |
  | 3  | 11/15/2022 | Yash Patil | 113.14 | ... |
  | 4  | 11/06/2022 | Yiwei Jiang | 96.45 | DDQN |
  | 5  | 11/15/2022 | Aniket Patil | 92.18 | ... |
  | 6  | 11/14/2022 | Samarth Shah | 85.39 | DDQN with Prioritized Replay |
  | 7  | 11/15/2022 | Neet Mehulkumar Mehta | 80.39 | ... |
  | 8  | 11/15/2022 | Noopur Koshta | 79.68 | ... |  
  | 9  | 11/15/2022 | Kunal Nandanwar | 79.68 | ... |            
  | 10  | 11/14/2022 | Aadesh Varude | 71.65 | Vanilla DQN |
  | 11  | 11/1t/2022 | Rutwik Bonde | 69.52 | ... |  
  | 12  | 11/07/2022 | Brown, Galen | 69.01 | Basic DQP with reward shaping |
  | 13  | 11/5/2022  | Ryan Killea | 67.12 | ... |
  | 14  | 11/14/2022  | Rushabh Kheni | 65.51 | Vanilla DQN with Deepmind architecture |  
  | 15  | 10/30/2022 | Jack Ayvazian | 47.49 | Double DQN, DeepMind Architecture |  
  


## Installation
Type the following command to install OpenAI Gym Atari environment in your **virutal environment**.

`pip install opencv-python-headless gym==0.25.2 gym[atari] autorom[accept-rom-license]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python test.py --test_dqn`

## Goal
In this project, you will be asked to implement DQN to play [Breakout](https://www.gymlibrary.dev/environments/atari/breakout/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/). The goal of your training is to get averaging reward in 100 episodes over **40 points** in **Breakout**, with OpenAI's Atari wrapper & unclipped reward. For more details, please see the [slides](https://docs.google.com/presentation/d/1jQ1mvFWxpoPJMebTxct-PDBzmGT3-HkVfUnZOzOAFsA/edit?usp=sharing).

<img src="/Project3/materials/project3.png" width="80%" >


## Hints
* [Naive Pytorch Tutorial](https://github.com/yingxue-zhang/DS595CS525-RL-Projects/blob/master/Project3/materials/Pytorch_tutorial.ipynb)
* [How to Save Model with Pytorch](https://github.com/yingxue-zhang/DS595CS525-RL-Projects/blob/master/Project3/materials/How%20to%20Save%20Model%20with%20Pytorch.pdf)
* [Official Pytorch Tutorial](https://pytorch.org/tutorials/)
* [Official DQN Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [Official DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)
* [DQN Tutorial on Medium](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)

## Tips for Using GPU on Google Cloud
* [How to use Google Cloud Platform](https://docs.google.com/document/d/1JfIG_yBi-xEIdT6KP1-eUpgLDoY3t2QrAKULB9yf01Q/edit?usp=sharing)
* [How to use Pytorch on GPU](https://docs.google.com/document/d/1i8YawKjEwg7qpfo7Io4C_FvSYiZxZjWMLkqHfcZMmaI/edit?usp=sharing)
* Other choice for GPU
  * Use your own GPU
  * Apply [Ace account](https://arc.wpi.edu/computing/accounts/) or[Turing account](https://arc.wpi.edu/computing/accounts/) from WPI 
  
