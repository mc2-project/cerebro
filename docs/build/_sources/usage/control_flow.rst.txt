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

Currently, cond can only have a single conditional operator.







For-Loops
################

Currently, Cerebro uses the SCALE-MAMBA @for_range construct when using for-loops. 
No extra work needs to be done on the developer's end, one can just write::
	for i in range(n)
and have that code be transformed into a representation the underlying framework understands.

There are a few limitations on the for-loop construct (which can be resolved by unrolling the loop)
	* Assignments to variables outside the for-loop scope cannot be made.
	* 







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
