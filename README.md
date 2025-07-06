# Ultimate Tic Tac Toe - Master Thesis Project

This repository contains the code and resources developed for my Master’s Thesis on **Ultimate Tic Tac Toe (UTTT)**, a complex extension of the classic Tic Tac Toe game. The project explores the application of **Reinforcement Learning (RL)** techniques to UTTT, leveraging its unique combination of local and global strategic dynamics as an ideal testbed for evaluating modern RL algorithms in large, structured action and state spaces.

After introducing the theoretical foundations of reinforcement learning, game theory, and algorithmic strategy, the study focuses on the design, training, and evaluation of several RL agents, including:

- **Deep Q-Networks (DQN)**
- **Double DQN (DDQN)**
- **Advantage Actor-Critic (A2C)**
- **Proximal Policy Optimization (PPO)**

The project implements a dedicated environment and training pipeline for self-play reinforcement learning. Agents are first trained in isolation and evaluated against random agents to establish performance baselines. Advanced techniques such as residual neural networks, data augmentation, and hyperparameter tuning are applied to refine the most promising models.

A key contribution of this work is the integration of **best response training** to evaluate the exploitability of learned policies. By training a separate best response agent using a DQN-based framework, this study identifies vulnerabilities in self-play agents and highlights the need for population-based training approaches to approximate **Nash equilibria**.

The final DDQN-based model, enhanced with residual architectures and rotational data augmentation, achieves an **85% win-rate** against all other trained agents, corresponding to a **72% relative improvement** over the original baseline.

This thesis demonstrates the effectiveness of reinforcement learning in mastering a non-trivial board game from scratch while also exposing the limitations of standard self-play in producing robust strategies. It provides a reproducible framework combining insights from RL, neural network design, and game theory for developing competitive agents in discrete, turn-based games with high strategic complexity.


👉 [Try the Ultimate Tic Tac Toe demo here](https://uttt-unipd.vercel.app)  

Play directly in your browser against the AI agents developed in this project.
