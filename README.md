<div align="center">

# ESP-Dataset (ICRA 2024)

#### ESP: Extro-Spective Prediction for Long-term Behavior Reasoning in Emergency Scenarios

 ### [Introduction](#introduction) | [Video](#video) | [News](#news) | [Dataset](#dataset) | [Download](#download) | [Website](https://esp-dataset.github.io/) | [Paper](https://arxiv.org/pdf/2405.04100)
</div>

## Introduction
Emergent-scene safety is the key milestone for fully autonomous driving, and reliable on-time prediction is essential to maintain safety in emergency scenarios. However, these emergency scenarios are long-tailed and hard to collect, which restricts the system from getting reliable predictions. In this paper, we build a new dataset, which aims at the long-term prediction with the inconspicuous state variation in history for the emergency event, named the Extro-Spective Prediction (ESP) problem.

- The ESP-Dataset with semantic environment information is collected over 2k+ kilometers focusing on emergency-event-based challenging scenarios. 

- A new metric named CTE is proposed for comprehensive evaluation of prediction performance in time-sensitive emergency scenarios. 
  
- ESP feature extraction and network encoder are introduced, which can be used to enhance existing backbones/algorithms seamlessly.

## Video

Click the following Graphical Abstract for the illustration video!

<div align="center">

[![A teaser of ESP datset](sources/graphical_abstract.png)](https://youtu.be/SsYsCZs2Sys?si=iXvVQ3KHNAICLutS)
</div>

## News
- [**Jul 24, 2024**] The full dataset is released.
- [**Jun 10, 2024**] A mini split of the dataset is released.


## Dataset
The dataset structure of tokens is shown below:
```bash
tokens/
├── train/
│   ├── token1/
│   ├── token2/
│   └── ...
├── val/
│   ├── token1/
│   ├── token2/
│   └── ...
└── test/
    ├── token1/
    ├── token2/
    └── ...
```

The dataset structure of tokens_by_mons is shown below:
```bash
tokens_by_mons/
├── mon1/
│   ├── token1/
│   ├── token2/
│   └── ...
├── mon2/
│   ├── token1/
│   ├── token2/
│   └── ...
└── ...
```

For each samplem, the structure is shown as below:
```bash
token
├── MomentId
├── Timestamp
├── TokenId
├── MapId
├── SceneInformation
│   ├── lane_type
│   ├── road_type
│   ├── time_of_day
│   ├── weather_conditions
│   └── ...
├── SemanticInfrastructure
│   ├── speed_monitor
│   ├── near_junction
│   ├── rare_road_objects
│   └── ...
├── EgoVehicleInformation
│   ├── vehicle_id
│   ├── vehicle_type
│   └── ...
├── TvInformation
│   ├── vehicle_id
│   ├── vehicle_type
│   └── ...
├── OtherVehiclesInformation
│   ├── vehicle1
│   ├── vehicle2
│   └── ...
└── ExtroSpectivePredictionFeatures
    ├── tv_dist_to_ev
    ├── tv_speed_to_ev
    └── ...

```


## Download
This section provides a link to the Mini Split and the full version of ESP-Dataset:

[Download ESP-Dataset Mini Split](https://drive.google.com/file/d/1LFtYyoKmPdx7luJsO5WhJFSwhg1jh9qd/view?usp=sharing)

The full dataset contains three separate files: "tokens," "tokens_by_mons." The "tokens_by_mons" file contains samples arranged by their respective moments, while the "tokens" file contains samples randomly grouped together. However, in terms of total samples, "tokens" and "tokens_by_mons" are equivalent.

[Download ESP-Dataset Full Dataset](https://drive.google.com/drive/folders/1Yhv7y7owlYQ2bJF1m56iqPgsQKvEj0Ik?usp=sharing) 

[Download ESP-Dataset Maps](https://huggingface.co/datasets/Etadingrui/ESP) 


## Citation
If using our data in your research work, please cite the following paper:
```
@article{dingrui2024esp,
      author    = {Wang, Dingrui and Lai, Zheyuan and Li, Yuda and Wu, Yi and Ma, Yuexin and Betz, Johannes and Yang, Ruigang and Li, Wei},
      title     = {ESP: Extro-Spective Prediction for Long-term Behavior Reasoning in Emergency Scenarios},
      journal   = {2024 IEEE International Conference on Robotics and Automation (ICRA)},
      year      = {2024},
    }
```
