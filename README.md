MC2: A Coopetitive Learning Platform

### Installing emp-agmpc (semi-honest version)

1. Create a directory for EMP toolkit.

2. Run the following script to install the prerequisite:
`wget https://goo.gl/wmt4KB -O install.sh && bash install.sh`

3. `cd emp-agmpc && cmake . && make`

### Install SCALE-MAMBA (semi-honest version)

1. Follow the instruction in this PDF file:  https://github.com/wzheng/mc2/blob/master/SCALE-MAMBA/Documentation/Documentation.pdf

### Compilation

1. If you wish to compile arithmetic circuit, execute  `cd SCALE-MAMBA; python compile.py a Programs/app_directory/` to compile.
If you wish to compile a boolean circuit, execute `cd emp-agmpc; python compile.py b Programs/app_directory`. 

2. The compilation step will generate the appropriate circuit files in `Programs/app_directory/`

### Data I/O

For now, MC2 supports file input and output for both SPDZ and AG-MPC.

1. To generate input data, cd to Input_Data and run `gen_data.py`. You should put the appropriate input data into this script. This python script will generate input files for both SPDZ and AG-MPC.

2. Currently, to get output from SPDZ/AG-MPC, you will need to call `reveal_all` inside the `.mpc` files.

3. To parse the AG-MPC output, cd into the Output_Data directory and run `agmpc_output_parser.py`. This script will ask for the directory name of the circuit (e.g., `Programs/app_directory/`).