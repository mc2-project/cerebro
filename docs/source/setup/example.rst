.. _example:

**********************************
Millionaire's Problem Example
**********************************


MPC Program
*************
We can walk through a simple  example of MPC which is called the Millionaire's problem. The premise is that there are 2 millionaries Alice and Bob, each with a certain net worth. The Millionare's problem is to determine who is the richest without evealing any information about any individual millionaire's net worth.

At the top of the program, we can declare our constants:
::
	ALICE = 0
	BOB = 1

We then read in our input data:
::
	alice_networth = sint.read_input(ALICE)
	bob_networth = sint.read_input(BOB)

We then compare the two player's net worth:
::
	if alice_networth < bob_networth:
		res = BOB
	else:
		res = ALICE

We can then reveal the final result:
::
	reveal_all(res, "The player who is richest")





