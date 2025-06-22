# DexGraspMotionChallenge2025

## Overview

This repository provides example code for training and testing on grasping trajectories of a single object. It demonstrates how to set up the training pipeline and evaluate the performance of learned policies within a simulated environment.

![Grasping Demo](assets/Grasping.gif)

## 1. Environment Setup

- Create a conda environment
  
  <pre><code>conda create -n DexGraspMotionChallenge2025 python==3.8
  conda activate DexGraspMotionChallenge2025</code></pre>

- Install IsaacGym
  - Download IsaacGym
  - Extract the downloaded files to the main directory of the project
  - Use the following command to install IsaacGym
  <pre><code>pip install -e ./isaacgym/python</code></pre>
- Install PyTorch
  <pre><code>pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117</code></pre>
- Install PyTorch3D
  <pre><code>git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d
  pip install -e .</code></pre>
- Install other dependencies
  <pre><code>cd ..
  pip install -e .</code></pre>
  
## 2. Dataset Download

You can download the mesh data of objects in GraspM3 from the link.

The structure of the mesh data for a single object is as follows:

<pre><code>meshdata/
├── core-bottle-a02a1255256fbe01c9292f26f73f6538/
│   └── coacd/
│       ├── coacd.urdf
│       ├── coacd_1.urdf
│       ├── decomposed.obj
│       ├── decomposed.wrl
│       ├── decomposed_log.txt
│       ├── model.config
│       ├── coacd_convex_piece_0.obj
│       ├── coacd_convex_piece_1.obj
│       ├── coacd_convex_piece_2.obj
│       ├── coacd_convex_piece_3.obj
│       └── coacd_convex_piece_4.obj</code></pre>

You can download the GraspM3 dataset from the link.
The compressed package contains multiple `.npy` files, each named after the object ID.

Each `.npy` file is a dictionary with the following keys:

- `obj_rotmat`: (B, 3, 3) array of object rotation matrices.
- `obj_scale`: (B,) array of object scaling factors.
- `grasp_seqs`: (B, T, D) array representing grasp trajectories.
  
Here, B is the number of trajectories, T is the sequence length, and D = 28 is the dimension of each grasp step, consisting of:
- the first 3 dimensions: global translation of the hand,
- the next 3 dimensions: global rotation of the hand,
- the remaining 22 dimensions: joint angles of the hand.

The illustration of the initial pose of the dexterous hand is shown below.

![Image](assets/Image.png)
  
## 3. Training and Testing Examples

We provide two training approaches based on different learning frameworks.Both methods utilize [DexRep](https://arxiv.org/pdf/2303.09806), a representation for dexterous grasping that encodes both geometric and spatial hand-object information. DexRep consists of three components: (1) Occupancy Feature, (2) Surface Feature, and (3) Local-Geo Feature.

**Method 1: MLP + DexRep**

In this baseline, we use a multi-layer perceptron (MLP) as the policy network trained on top of DexRep features extracted from hand-object configurations.

**Method 2: DP3 + DexRep**

This approach is inspired by [3D diffusion policy](https://arxiv.org/abs/2403.03954).

We integrate DexRep as the feature representation within the DP3 diffusion-based policy learning framework.

### Training Example

**Method 1**

Run the training with:

<pre><code>cd dexgrasp
python train_bc_lighting_dexrep.py</code></pre>

**Method 2**

Run the training with:

<pre><code>cd dexgrasp
python train_bc_lighting_dp3_dexrep.py</code></pre>

### Testing Example

**Method 1**

Run the following command to perform testing:

<pre><code>python -u bc_env_infer.py --task=ShadowHandGraspDexRepIjrr --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --headless</code></pre>

**Method 2**

Run the following command to perform testing:

<pre><code>python -u bc_env_infer_multisteps.py --task=ShadowHandGraspDexRepIjrr --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --headless</code></pre>
