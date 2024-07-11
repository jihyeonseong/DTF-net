# Towards Dynamic Trend Filtering through Trend Point Detection with Reinforcement Learning (IJCAI24)
* This is the author code implements "Towards Dynamic Trend Filtering through Trend Point Detection with Reinforcement Learning," a paper accepted at IJCAI 2024.

## Overview
<img width="373" alt="스크린샷 2024-07-11 오전 11 39 38" src="https://github.com/jihyeonseong/DTF-net/assets/159874470/2b078f9a-e425-45bb-8ce8-b30f8dcb9cee">

Trend filtering simplifies complex time series data by applying smoothness to filter out noise while emphasizing proximity to the original data. However, existing trend filtering methods fail to reflect abrupt changes in the trend due to 'approximateness,' resulting in constant smoothness. This approximateness uniformly filters out the tail distribution of time series data, characterized by extreme values, including both abrupt changes and noise. In this paper, we propose Trend Point Detection formulated as a Markov Decision Process (MDP), a novel approach to identifying essential points that should be reflected in the trend, departing from approximations. We term these essential points as Dynamic Trend Points (DTPs) and extract trends by interpolating them. To identify DTPs, we utilize Reinforcement Learning (RL) within a discrete action space and a forecasting sum-of-squares loss function as a reward, referred to as the Dynamic Trend Filtering network (DTF-net). DTF-net integrates flexible noise filtering, preserving critical original subsequences while removing noise as required for other subsequences. We demonstrate that DTF-net excels at capturing abrupt changes compared to other trend filtering algorithms and enhances forecasting performance, as abrupt changes are predicted rather than smoothed out.

<img width="877" alt="스크린샷 2024-07-11 오전 11 40 17" src="https://github.com/jihyeonseong/DTF-net/assets/159874470/3185c8bb-514d-4166-90c3-80d6afd7fbac">

* We identified the issue of `approximateness,' which leads to constant smoothness in traditional trend filtering, filtering out both abrupt changes and noise.
* We introduce Trend Point Detection formulated as an MDP, aiming to identify essential trend points that should be reflected in the trend, including abrupt changes. Additionally, we propose DTF-net, an RL algorithm that predicts DTPs through agents.
* We employ the forecasting sum-of-squares cost function, inspired by reward function learning based on GP, which allows for the consideration of temporal dependencies when capturing DTPs. A sampling method is applied to prevent the overfitting issue.
* We demonstrate that DTF-net excels at capturing abrupt changes compared to other trend filtering methods and enhances performance in forecasting tasks.

## Running the codes
### STEP 1. Download the benchmark datasets for time series forecasting
* The datasets can be downloaded form the [BasicTS raw data repository](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp).
* Create a directory named "data" and store downloaded datasets within it.
### STEP 2. Train the RL agent to extract the Dynamic Trends and get forecasting results.
```
python main.py
```
You can set your hyperparameters as you want!

## Citation
```
@misc{seong2024dynamictrendfilteringtrend,
      title={Towards Dynamic Trend Filtering through Trend Point Detection with Reinforcement Learning}, 
      author={Jihyeon Seong and Sekwang Oh and Jaesik Choi},
      year={2024},
      eprint={2406.03665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.03665}, 
}
```
