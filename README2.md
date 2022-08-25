This code is based on ...

To run the `experminent/train.py`:
on MacOS:
- download `mujoco200` from [here](http://www.roboti.us/download.html) and move the folder to `data`.
- Eventhough Roboti LLC provides an unlocked version of MuJoCo but we still need to download the activation key `mjkey.txt` from [here](http://www.roboti.us/license.html) and move it inside `mujoco200/bin`.
- Before running the requirements, install gcc-9 using brew, brew install gcc@9, run the following command,export CC=/usr/local/bin/gcc-9
- run conda env create -n DRL22_python3.7
- conda install --yes --file requirements.txt
- conda install -c conda-forge mpi4py mpich
- conda install matplotlib==3.0.1