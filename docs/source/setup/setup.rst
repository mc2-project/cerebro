*************************
Cerebro Setup
*************************

Cerebro has a few core dependencies that need to be setup before it can be used.

SCALE-MAMBA
################

The documentation fully detailing SCALE-MAMBA can be obtained by running ``make doc`` in the SCALE-MAMBA repository.

The following is copied from SCALE-MAMBA's current setup documentation.

Prerequisite Libraries:
*************************
	* gcc/g++, tested with version 7.2.1
	* MPIR (compiled with the -cxx flag)
	* python 2.7.5 (ideally with gmpy2 installed)
	* OpenSSL (tested with version 1.1.0)
	* Crypto ++ (tested with version 7.0)



Setup Steps for SCALE-MAMBA
*****************************
	* After installing the libraries, go to CONFIG.mine and change the entry corresponding with ``ROOT`` to point to the current path of where the SCALE-MAMBA directory is. Also, change the entry corresponding with ``OSSL`` to have it point to the installed OpenSSL directory.
	* Run ``make progs`` and SCALE-MAMBA should compile.
	* Create a config file with all the servers participating in the multiparty computation. Line i should have the IP address of the i'th server.
	* Go to mc2/Config and run the script gen_cert.py and pass in the previously made config file along with the number of parties participating.
	* Additionally, run ``python mc2/Config/make_config.py`` which sets up the Network configuration for SCALE-MAMBA to use.
	* Then in mc2/SCALE-MAMBA, run the command: ``echo '2 1 p' > ./Setup.x`` where p is the length of the prime in terms of number of bits used in the secure computation.


EMP-AGMPC
################
Prerequisite Libraries:
*************************
	* emp-toolkit:
		* Install: 
			* cmake 
			* git 
			* build-essential 
			* libssl-dev 
			* libgmp-dev
			* Boost
			* relic
		* cd into emp-tool and run ``cmake . && make && sudo make install``.
	* emp-ot: Installation instructions are here: https://github.com/emp-toolkit/emp-ot
		* After installing emp-toolkit, just cd into emp-ot and run ``cmake . && make && sudo make install``.
			
Setup Steps for EMP-AGMPC
***************************
	* The make_config.py script should have set up all the network parameters inside emp-agmpc/emp-agmpc/cmpc_config.h, but if not, you'll have to manually input the IP's of each server.
		* IP[i] will hold the IP address of the ith server.
		* Note that everything here is 1-indexed so the first server goes into IP[1].
	* Run ``cmake . && make``


Alternatives
***************************
	* Alternatively, you can pull the docker image from here: ``docker pull rdeng2614/cerebro:initial_image`` which has all the dependencies installed. Then, you can clone the cerebro repository and directly start setting up from there.



Testing
***************************
	* Included with Cerebro are a set of tests for both SCALE-MAMBA and emp-agmpc.
	* To run the tests with SCALE-MAMBA, run ``cd mc2/crypto_backend/SCALE-MAMBA`` and then run ``python test_scripts/test_scale_mamba.py``.
	* To run the tests with emp-agmpc, run ``cd mc2/crypto_backend/emp-toolkit/emp-agmpc`` and then run ``python test_scripts/test_gc.py``.

