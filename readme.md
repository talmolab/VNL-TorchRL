# VNL-TorchRL with CPU Mujoco
The CPU Mujoco Version VNL implementation with TorchRL as the deep learning backend

## Instruction for our modified CPU Mujoco
1. Building Mujoco from source (https://mujoco.readthedocs.io/en/stable/programming/index.html)
    1. Install cmake by `sudo apt install cmake`
    2. Clone Emil’s modified Mujoco repository from GitHub with branch of “feature/batched-render” (https://github.com/emiwar/mujoco/tree/feature/batched-render )
    3. Create a new build & release directory and cd into it
    4. Run `cmake ../mujoco` to get all dependency and ready to build (check for dependnecy)
        - Run `apt-cache search <packages>` to find the version
        - Run `sudo apt install <package>` to install <packages>
    5. Run `cmake --build .` to actually build the package
    6. Export the build version: `cmake ../mujoco -DCMAKE_INSTALL_PREFIX=../release`
    7. After building, install with `cmake --install`
    8. Run `cd release`, you should see normal Mujoco C++ version now

2. Building python packages from compiled C++ code (When need to reflect changes in the system, only do this step is necessary)
    1. In the root file, create a new venv with `python -m venv cpu_mujoco`
    2. Run `source cpu_mujoco/bin/activate` to activate the virtual environment
    3. Run `cd mujoco/python`
    4. Run `bash make_sdist.sh` to make the python package
    5. Run `cd dict` and the python package `mujoco-3.1.4.tar.gz` should be there
        - Set env flag with `export MUJOCO_PLUGIN_PATH=~/release`
        - Set env flag with `export MUJOCO_PATH=~/release`
    6. Run `pip install mujoco-3.1.4.tar.gz` to install the python package that can be used just like the normal Mujoco python binding version.
