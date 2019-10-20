.. _control_flow:

**************************************************
Cerebro's supported control flow
**************************************************



Conditionals
################

Cerebro currently supports basic conditionals and assignment operators under those conditionals.
It transforms such assignment operators into a compound statement that erases the use of conditionals.

For example, if we have the following conditional:
::
	if cond:
		x = 5
	else:
		x = 10

Then, Cerebro transforms it into:
::
	x = cond * 5 + (1-cond) * 10

Limitations: 
	* Condition can only have a single conditional operator.
	* Currently, only assignment operators are supported in the body of the conditional.







For-Loops
################

Currently, Cerebro uses the SCALE-MAMBA @for_range construct when using for-loops. 
No extra work needs to be done on the developer's end, one can just write::
	for i in range(n)
and have that code be transformed into a representation the underlying framework understands.

Limitations: (which can be resolved by unrolling the loop)
	* Assignments to variables outside the for-loop scope cannot be made.








Loop-Unrolling
**********************

Loop unrolling is currently a work-in progress, but it allows code within a for-loop to be unrolled into a series of assignment statements and function calls.

After enabling loop unrolling, the following example code:
::
	x = 0
	for i in range(2):
		x += i
would be transformed into:
::
	x = 0
	i = 0
	x += i
	i = 1
	x += i

It allows more flexibility in what to code in for-loops for the developer.


Function Inlining
**********************
Function inlining is also a work in progress. It replaces function calls with the entire inlined version of the function call.
For example, if we have the following code:
::
	def f(y):
		x = 5
		print("Hello")
		return x + y

	def g(x):
		return f(x)

	g(3)

After function inlining, it would be transformed into:
::
	g_x = 3
	f_y = 3
	f_x = 5
	print("Hello")
	f_ret = f_x + f_y
	g_ret = f_ret


Limitations:
	* Currently function inlining is a bit wonky dealing with OOP.


