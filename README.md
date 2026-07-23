# dynamic-sim-tutorial
This repository contains example code used to teach dynamic simulation fundamentals. This code was written for readability, not efficiency.

## Running the code

1. Install uv if necessary.
    ```shell
    sudo apt update
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```
2. Run the following in the base directory to create the `/results` folder.
    Plots and animations (in .gif format) will automatically be deposited there.
    ```shell
    mkdir results
    ```
3. Run your desired script in the following way.
    ```shell
    uv run src/cube_3d_con_tstep.py
    ```
