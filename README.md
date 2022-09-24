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
    - `pip install matplotlib==3.0.1`
- Change in util.py and train.py where For macOS is written.


### Getting started (on Ubuntu):
- Create a virtual environment using Python3.7: `python3.7 -m venv ~/venv/RCHAC`
- Run `pip3 install -r requirements_gpu.txt` if you have GPU, otherwise `pip3 install -r requirements.txt`

### To download the data:
- If you're using macOS, ignore the next two bullet points. Otherwise, delete `data/mujoco200`and follow the following instructions.
- download `mujoco200` from [here](http://www.roboti.us/download.html) and move the folder inside the `data`.
- Even though Roboti LLC provides an unlocked version of MuJoCo but we still need to download the activation key `mjkey.txt` from [here](http://www.roboti.us/license.html) and move it inside `mujoco200/bin`.

### Command line options
- To run the `experminent/train.py`: 
  `python3.7 experiment/train.py--env AntFourRoomsEnv-v0 --n_epochs 100 --fw 1 --eta 0.5 --regularization 1`
  - `--fw 1` indicates if curiosity-based rewards should be used.
  - `--eta 0.75` indicates the balance between using intrinsic rewards and external rewards.
  - `--regularization 1` indicates using regularization of the subgoal learning.
  - `--num_cpu` indicates the number of CPU cores to be used for MPI.
- To run a trained policy or to continue its training:
    `python3.7 experiment/train.py--env AntFourRoomsEnv-v0 --restore_policy path-to-the-policy --n_epochs 100 --fw 1 --eta 0.5 --regularization 1`

### Visualization tool: 
- To see the visualization of the different runs of this [project](https://wandb.ai/rfarah/RCHAC?workspace=user-rfarah) on W&B.
- To use tensorboard, which is used from the authors of CHAC, run the following:
`tensorboard --logdir=./data/d4a6886/AntReacherEnv-v0 --host localhost --port 8088` or 
`tensorboard --logdir=./data/eef7a77/UR5ReacherEnv-v1 --host localhost --port 8088`
