## Brief Overview on how to use SPDZ-2 ADMM ##
This was based on the now deprecated SPDZ-2 Library.

admm.mpc

* Very naive implementation of admm, no vectorization
* Currently compiles on SPDZ-2, but might not on the other frameworks, so may have to fix.
* 


admm_client.cpp
* In the Makefile, after externalIO, add admm-client.x
* Similarly, near the bottom define admm-client.x by copying what they have for bankers-bonus in the Makefile.
* Use: ./admm-client json_file_name nparties num_rows num_cols
* In the main function, there is an array of strings called `host_names` which contains the host names of all n parties.


gen_data.py
* Used to generate fake data.
* Use: python3 gen_data.py nparties num_rows num_cols
* Output: Generates nparties json files with the fake data, pairs well with the admm client as of right now.


Right now, the client reads in the data, and sends it to the SPDZ server which accepts the client's input and then runs the secure computation. 
The function `accept_client_input` accepts the client's input. After reading the input, `admm` is called to run ADMM.
