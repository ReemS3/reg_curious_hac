## Regularized Curiosity Hierarchical Reinforcement Learning

This code is based on Curious Hierarchical Actor Critic reinforcement learning [CHAC](https://github.com/knowledgetechnologyuhh/goal_conditioned_RL_baselines) and active Hierarchical Exploration
with Stable Subgoal representation learning [HESS](https://github.com/SiyuanLee/HESS/tree/8f93124533ac07037172b8dd99080c17fef0a961).


### Known issues:
This code works with Python3.7 on MacOS

### Getting started (on macOS):
- Before installing the requirements, run the following commands:
    - `install gcc-9 using brew`
    - `brew install gcc@9`
    - `export CC=/usr/local/bin/gcc-9` 
- Create a virtual environment using Python3.7: `run conda env create -n RCHAC`
- Install the requirements: `conda install --yes --file requirements.txt`
- We couldn't download all the required libraries using conda, thus we used pip as follows:
    - `conda install -c conda-forge mpi4py mpich`
    - `conda install matplotlib==3.0.1`

### To download the data:
- If you're using macOS, ignore the next two bullet points. Otherwise, delete `data/mujoco200`and follow the following instructions.
- download `mujoco200` from [here](http://www.roboti.us/download.html) and move the folder inside the `data`.
- Even though Roboti LLC provides an unlocked version of MuJoCo but we still need to download the activation key `mjkey.txt` from [here](http://www.roboti.us/license.html) and move it inside `mujoco200/bin`.

### Command line options
To run the `experminent/train.py`:

### Visualization tool: 
To see the visualization of the different runs of this [project](https://wandb.ai/rfarah/RCHAC?workspace=user-rfarah) on W&B.