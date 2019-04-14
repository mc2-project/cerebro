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

## Machine Configuration

Below is how Weikeng loads the machine configuration.

### Add environment variables indicating the party ID
For the convenience of benchmarking, I set the party ID for each party in their `~/.bashrc` file. 

Note that SCALE-MAMBA's party-ID is zero-based, but AG-MPC's party-ID is one-based. Thus, to set the party ID for the first party, we run:
```
echo "MY_PARTY_ID_GC=1" >> ~/.bashrc && echo "MY_PARTY_ID_A=0" >> ~/.bashrc
source ~/.bashrc
```

### Change the network speed
We use `tc` to change the upload speed and download speed. The script applies the speed limit to parties with IP address in `172.31.0.0/16`. If you are testing on the Internet, probably you don't need to manually set the network speed.
```
cd ~/mc2/
git pull
cd Config
./setup.sh 100Mbit 100Mbit
```

### Set the machine IP addresses
Create your own machine IP file like `Config/wk.txt`. You might want to name it `Config/zheng.txt` (so the first letter is different and easier to tab) or `Config/ryan.txt`. 

In this file, list the IP addresses in the order of party IDs. For example, if you have three parties:
```
172.31.39.112
172.31.41.155
172.31.44.33
```
This implies that `172.31.39.112` is the first party.

### Run SCALE-MAMBA on test_dt for four parties
Below is the code for running SCALE-MAMBA on `test_dt` for four parties. If you want to run on different number of parties, change the line 2, where the last number indicates the number of parties in this computation. If you want to run different types of algorithms, change line 5 and line 8 (for compiling and input generation).
```
cd ~/mc2/
python Config/make_config.py Config/wk.txt 4
cd SCALE-MAMBA
make progs
python compile.py a Programs/test_dt/
cd ..
cd Input_Data/
python gen_data_dt.py .
cd ../
cd SCALE-MAMBA
time ./Player.x $MY_PARTY_ID_A Programs/test_dt
echo "ok!"
```

### Run AG-MPC on test_dt for four parties
Below is the code for running AG-MPC on `test_dt` for four parties. If you want to run on different number of parties, change the line 2, where the last number indicates the number of parties in this computation. If you want to run different types of algorithms, change line 3 and line 5 (for compiling and input generation).
```
cd ~/mc2/
python Config/make_config.py Config/wk.txt 4
python compile.py b Programs/test_dt/
cd Input_Data/
python gen_data_dt.py .
cd ../
cd emp-agmpc/
./bin/run_circuit $MY_PARTY_ID_GC 5000 Programs/test_dt/ Input_Data/ Output_Data/ s
echo "ok!"
```
