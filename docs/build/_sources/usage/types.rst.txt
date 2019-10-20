.. _types:

*************************
Types used in Cerebro
*************************



Clear Types
################

Clear types are used to store values that are visible to every party. Use these values to store global variables and constants which do not depend on any user's input.


**c_int**: A clear integer modulo ``p``, where ``p`` is the prime field. :
::
	x = c_int(5)

Main operations supported with clear integers
***************************************************
	* Addition
	* Subtraction
	* Multiplication
	* Comparisons
	* Shifting
	* Bit-decompose
	* Hash digest


**c_fix**: A clear fixed-point number modulo ``p``. :
::
	x = c_fix(2.5)

The precision must be specified at the top of the .mpc program as follows: ``Params.set_params(k=, f=)``, where ``k`` indicates the number of bits used to represent a floating point number and ``f`` represents the number of bits to represent the decimal places.

Main operations supported with clear fixed-point values
*********************************************************
	* Addition
	* Subtraction
	* Multiplication
	* Comparisons
	* Bit-decompose
	* Division

Additionally, Cerebro supports containers for clear integers and clear fixed-point values.

**c_int_mat**: A clear int matrix. :
::
	# mat is assigned to a 5 x 7 clear integer matrix
	mat = c_int_mat(5, 7)

**c_fix_mat**: A clear fixed-pointed matrix. :
::
	# mat is assigned to a 5x7 clear fixed-point matrix
	mat = c_fix_mat(5, 7)


List of operations supported with clear-valued containers
************************************************************
The list of operations that be done on matrices can be found in the :ref:`Cerebreo mllib section <mllib>`. 





Secret Types
################

Secret types are used to store values that are not visible to any party. Use these values to store values that depend on individual party's inputs.

**s_int**: A clear integer modulo ``p``, where ``p`` is the prime field. :
::
	x = s_int(5)


List of operations supported with secret values
***************************************************
	* Addition
	* Subtraction
	* Multiplication
	* Comparisons
	* Shifting
	* Bit-decompose

**s_fix**: A clear fixed-point number modulo ``p``. :
::
	x = s_fix(2.5)

The precision must be specified at the top of the .mpc program as follows: ``Params.set_params(k=, f=)``, where ``k`` indicates the number of bits used to represent a floating point number and ``f`` represents the number of bits to represent the decimal places.

List of operations supported with secret fixed-point values
*************************************************************
	* Addition
	* Subtraction
	* Multiplication
	* Comparisons
	* Division


Additionally, Cerebro supports containers for clear integers and clear fixed-point values.

**s_int_mat**: A clear int matrix. :
::
	# mat is assigned to a 5 x 7 secret integer matrix
	mat = s_int_mat(5, 7)

**s_fix_mat**: A clear fixed-pointed matrix. :
::
	# mat is assigned to a 5x7 secret fixed-point matrix
	mat = s_fix_mat(5, 7)


List of operations supported with secret-valued containers
************************************************************
The list of operations that be done on matrices can be found in the :ref:`Cerebreo mllib section <mllib>`. 
