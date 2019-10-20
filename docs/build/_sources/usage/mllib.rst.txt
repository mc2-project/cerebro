.. _mllib:

*************************
Cerebro's ML Library
*************************



Matrix Operations
###################

Currently, Cerebro supports a number of operations involving matrices.


**matadd(A, B)**: Element-wise addition of two matrices A and B.

**matsub(A, B)**: Element-wise subtraction of two matrices A and B.

**matmul(A, B)**: Matrix multiplication of matrices A and B.

**transpose(A)**: Transpose of matrix A.

**matinv(A)**: Inverse of matrix A.









Other useful functions or classes used in ML
################################################


**sigmoid(v)**: Performs the sigmoid function on value v.

**mat_const_mul(c, A)**: Multiplies each entry in matrix A by a constant value c.

**mat_assign(dest, src)**: Assigns dest matrix to src matrix. 

**get_identity_matrix(n)**: Returns an identity matrix of dimension n.

**class Piecewise**: Defines a piecewise function based on a sequence of boundaries. 

	* add_boundary(self, lower, upper, a, b)
		* Adds a linear function y = ax + b within the boundary [lower, upper]

Example of a ReLU function::

		# init a piecewise function with 2 boundaries (-infty, 0), (0, infty)
		relu = Piecewise(2)
		# Add a linear function for each boundary: y = ax + b
		relu.add(-float("inf"), 0, 0, 0)
		relu.add(0, float("inf"), 1, 0)


