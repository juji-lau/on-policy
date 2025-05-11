## Exploring Reward Sharing Strategies for Effective Cooperative Multi-Agent Task Completion
Juji Lau, Sanjana Nandi 

### Purpose
We modified the [official MAPPO implementation](https://github.com/marlbenchmark/on-policy/) to test the effects of different reward schemes - individual, partially shared, and shared on the learning dynamics and group behavior of agents in a Simple Spread environment. 

### Installation and Running the Code
To run this project, download and open `shared_rewards_proj.ipynb` in Google Colab. **Do not try to run the `shared_rewards_proj.ipynb` in on-policy.** 
Running the setup cells will clone this repository into your personal drive at the top level (MyDrive/on-policy) and allow you to run the rest of the cells. They will create a runs folder at the top level of your personal drive (MyDrive/runs) where the results will be stored.

### Citations
Yu, Chao, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu. "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games." arXiv:2103.01955, 2021.
