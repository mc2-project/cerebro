*************************
Cerebro Tutorial
*************************

While Cerebro can choose between arithmetic and boolean protocols, the process of automating it is currently not supported. The documentation below will detail how to run .mpc programs for both frameworks.

First step is writing an .mpc program.

Example on how to write .mpc programs can be found below:
	* :ref:`Cerebro examples <example>`


Arithmetic: SCALE-MAMBA
*****************************
	* First, write a .mpc program. There are examples in the mc2/Programs directory.
	* Compile the .mpc program by calling "./compile.py a [Program_Name]"
	* If the compilation is complete, then on each server run: "./Player [player_number] [Programs/program_name]"
		* player_number is 0-indexed and must match with the NetworkData.txt file in SCALE-MAMBA.



Boolean: EMP_AGMPC
*****************************
	* Write a .mpc program. There are examples in the mc2/Programs directory.
	* Compile the .mpc program by calling "./compile.py b [Program_Name]"
	* After compiling, in "Programs/[Program_Name]" there should be a circuit file called "agmpc.txt" and an input format file. Make sure to keep track of what directory they are in.
	* For boolean circuits, on each server run: ./bin/run_circuit [party_number] [port_number] [circuit_file_folder] [input_file_folder] [output_file_folder]
		* party_number here is 1-indexed and must match with the IP addresses in cmpc_config.h
		* output_file_folder is the directory in which Cerebro puts the output file which contains all output of the secure computation.
		* input_file_folder is the directory that contains the user input, more specifically the file "agmpc.input". This is typically generated using Cerebro's provided script shown below. 
		* circuit_file_folder is the directory that contains the circuit file: "agmpc.txt" and the input format file "agmpc.txt.input".

	* After the computation is run, the output file "agmpc.output" will be stored in the specified output_file_folder. 



Side Note: Input Data
*****************************
Arithmetic and Boolean circuits are able to read in user data. Currently, Cerebro has it set up such that the input data comes from a directory called Input_Data. In there, there is a script called ``gen_data.py`` which will convert data generated from another python file to input_data recognizable by Cerebro.

To have a user provide input data do the following:
	* In another file (for example hello.py), run the program that generates the user data.
	* At the very end of the program, set the variable ``data`` equal to the array of the user's input data. For example, if my input data is 1,2,3, then at the very end of the program we have the line: ``data = [1, 2, 3]``
	* Call ``python gen_data.py [file_name]`` where file_name is the python file containing the data.
	* The script generates a file f0 and input.txt. The former is used for arithmetic circuits and can be left alone while the latter text file can be moved depending on your needs. (More is explained above in the EMP_AGMPC section).

Limitations: Currently the way the script is written, the input size is at most 64 bits in length.


