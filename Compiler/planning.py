import math
from itertools import combinations
import program
import json


# --------------- SPDZ OFFLINE --------------- #
# Constants that
TRIPLES_PER_BATCH = 32768
BATCH_SIZE = 80

def get_constants_dict(constants_file):
	with open(constants_file, "r") as f:
		constants_dict = json.load(f)
	return constants_dict


def spdz_linear_cost(req_num, constants_dict):
	num_triples = req_num[('modp', 'triple')]
	num_bits = req_num[('modp', 'bit')]
	return spdz_linear_helper(num_triples + num_bits, constants_dict)



"""
def spdz_linear_helper(num_type, constants_dict):
	# All the constants have a '_l' to indicate that it belongs to the linear protocol.
	num_parties = constants_dict['P']
	total_cost = 0
	
	# c1 is the encryption cost for the triples
	total_cost += constants_dict['c1_l']

	# coordinator costs.
	coordinator_cost = 0
	# c2 captures the cost of adding (P-1) ciphertexts together
	coordinator_cost += constants_dict['c2_l']
	# c3 captures the cost of multiplying ciphertexts together
	coordinator_cost += constants_dict['c3_l']
	# c4 captures the network cost of sending and receiving the ciphertexts from the coordinator to all other parties.
	coordinator_cost += (constants_dict['c4_l']) * (num_parties - 1)


	total_cost += coordinator_cost/num_parties

	total_cost *= num_type

	# c is the initialization cost
	total_cost += constants_dict['c_l']

	return total_cost
"""



# Weikeng's version.
def spdz_linear_helper(num_type, constants_dict):
	# Linear currently from Weikeng's implementation is in the form of cost = mx + b, where the dependent variable x = 1/P and m = C1 + C2, and b = C3-C2
	# The slope "captures" the inverse relationship with the dependent variable x
	# Matching the google spreadsheet.
	batch_size = constants_dict['N']
	num_parties = constants_dict['P']
	slope = constants_dict['m_l']
	intercept = constants_dict['b_l']
	cost_per_triple = slope * num_parties + intercept
	print "Linear cost per triple: ", cost_per_triple
	#total_cost = math.ceil(num_type * 1.0 / batch_size) * cost_per_batch
	total_cost = num_type * cost_per_triple
	return total_cost



def spdz_quadratic_cost(req_num, constants_dict):
	num_triples = req_num[('modp', 'triple')]
	num_bits = req_num[('modp', 'bit')]

	return spdz_quadratic_helper(num_triples, constants_dict) + spdz_quadratic_helper(num_bits, constants_dict) + vectorized_triple_cost(req_num, constants_dict)


# Weikeng's version which combines everything into a slope and intercept.
def spdz_quadratic_helper(num_type, constants_dict):
	# Quadratic from Weikeng's experiments is cost = mx + b, where m = C2+C3, x = P-1, b = C1
	# Essentially the slope captures the linear relationship with the number of parties
	# Matching the google spreadsheet terminology
	batch_size = constants_dict['N']
	num_parties = constants_dict['P']
	slope = constants_dict['m_q']
	intercept = constants_dict['b_q']
	cost_per_triple = slope * num_parties + intercept
	print "Quadratic cost per triple: ", cost_per_triple
	cost_per_batch = batch_size * cost_per_triple
	total_cost = num_type * cost_per_triple
	# Add in vectorized cost.
	# TODO: Remember to subtract off the triples used for vectorization.
	# Currently sfix additions are acting a bit weird.
	return total_cost


"""
def spdz_quadratic_helper(num_type, constants_dict):
	num_parties = constants_dict['P']
	total_cost = 0 
	local_cost = 0
	# Encryption cost
	local_cost += constants_dict['c1_q']
	# Cost of sending ciphertext to all other parties
	local_cost += constants_dict['c2_q']
	total_cost += num_type * (num_parties - 1) * local_cost
	total_cost += constants_dict['c_q']
	return total_cost
"""


def flat_spdz_cost(req_num, constants_dict):
	try:
		linear_cost = spdz_linear_cost(req_num, constants_dict)
	except KeyError as e:
		linear_cost = float('inf')
		print "Key Error Linear!"
		print e

	try:
		quadratic_cost = spdz_quadratic_cost(req_num, constants_dict)
	except KeyError as e:
		quadratic_cost = float('inf')
		print "Key Error Quadratic!"
		print e

	d = dict()
	d['cost'] = min(linear_cost, quadratic_cost)
	if linear_cost < quadratic_cost:
		d['decision'] = 'linear'
		return d
	else:
		d['decision'] = 'quadratic'
		return d


def hierarchical_cost(req_num, constants_dict):
	try:
		num_parties = constants_dict['P']
		num_triples = req_num[('modp', 'triple')]
		num_bits = req_num[('modp', 'bit')]
		cost_triples, nl_triple, nr_triple = search_best_plan(num_parties, num_triples, constants_dict)
		cost_bits, nl_bit, nr_bit = search_best_plan(num_parties, num_bits, constants_dict)
		d = dict()
		d['cost'] = cost_triples + cost_bits
		d['nl_triple'] = nl_triple
		d['nr_triple'] = nr_triple
		d['nl_bit'] = nl_bit
		d['nr_right'] = nr_bit
		return d
	except KeyError as e:
		d = dict()
		d['cost'] = float('inf')
		d['nl_triple'] = ""
		d['nr_triple'] = ""
		d['nl_bit'] = ""
		d['nr_right'] = ""
		print "Key Error Hierarchical!!"
		print e
		return d

def search_best_plan(num_parties, num_type, constants_dict):
	num_per_batch = TRIPLES_PER_BATCH * BATCH_SIZE
	best_cost = float('inf')
	best_nl = -1
	best_nr = -1
	inter_bandwidth = constants_dict['inter_band']
	intra_bandwidth = constants_dict['intra_band']
	# Optimal number of parties in left region
	left_parties = constants_dict['left_parties']
	right_parties = constants_dict['right_parties']	
	p_l = left_parties
	p_r = num_parties - p_l

	num_possible_triples = [i * num_per_batch for i in range(int(math.ceil(1.0 * num_type / num_per_batch)))]
	for n_l in num_possible_triples:
		# Get how many triples n_r will be producing. Since we produce by the batch, need to take a ceiling.
		n_r = math.ceil(1.0 * (num_type - n_l)/num_per_batch) * num_per_batch
		L1 = max(1.0 * n_r * p_r * (p_l - 1) / p_l, 1.0 * n_l * p_l * (p_r - 1) / p_r)
		L2 = max(n_l * (p_l - 1), n_r * (p_r - 1))
		L3 = max(n_l * p_l, n_r * p_r)
		L4 = max(n_l, n_r)
		# Equation is: c1_h (L1 + L2) + c2_h * L3 + c3_h * L4 + c4_h * (L1 + L3)
		cost = constants_dict['c_h'] + constants_dict['c1_h'] * (L1 + L2) * intra_bandwidth + constants_dict['c2_h'] * L3 * inter_bandwidth + constants_dict['c3_h'] * L4 + constants_dict['c4_h'] * (L1 + L3)
		if cost < best_cost:
			cost = best_cost
			best_nl = n_l
			best_nr = n_r

	return best_cost, best_nl, best_nr



def spdz_offline_cost(req_num, constants_dict):
	d_flat = flat_spdz_cost(req_num, constants_dict)
	d_hierarchical = hierarchical_cost(req_num, constants_dict)
	if d_flat['cost'] == float('inf') and d_hierarchical['cost'] == float('inf'):
		print "Error, probably forgot to include some constants in the constants file."
		raise ValueError

	if d_flat['cost'] < d_hierarchical['cost']:
		flat_decision = d_flat['decision']
		d_flat['decision'] = 'flat_' + str(flat_decision)
		return d_flat

	else:
		d_hierarchical['decision'] = 'hierarchical'
		return d_hierarchical




def vectorized_triple_cost(req_num, constants_dict):
	# Currently doesn't seem to be a cost model, or fitted constants.
	# Vectorized triples are length 5-tuples
	num_parties = constants_dict['P']
	init_cost = constants_dict['c_v']
	total_cost = 0
	total_cost += init_cost
	# Based on the paper: c1 = f1 + f2 + g, c2 = g, c3 = f1
	c1 = constants_dict['c1_v']
	c2 = constants_dict['c2_v']
	c3 = constants_dict['c3_v']
	for k in req_num.keys():
		# For a vectorized triple follows this format: (left_rows, left_cols, right_rows, right_cols, type_of_matrix)
		if len(k) == 5:
			m = k[0]
			n = k[1]
			right_cols = int(k[3])
			one_vectorized_triple_cost = m * (num_parties - 1) * c1 + c2 * n * (num_parties - 1) + c3 * n
			total_cost += right_cols * one_vectorized_triple_cost


	print "Vectorized cost: ", total_cost

	return total_cost




# --------------- SPDZ Online Cost --------------- #
def spdz_online_cost(req_num, constants_dict):
	num_triples = req_num[('modp', 'triple')]
	num_bits = req_num[('modp', 'bit')]
	# Account for vectorized triples
	num_vectorized = 0
	print "Req num: ", req_num
	print "Keys: ", req_num.keys()


	
	for k in req_num.keys():
		if len(k) == 5:
			left_rows = k[0]
			left_cols = k[1]
			right_rows = k[2]
			right_cols = k[3]
			num_vectorized += left_rows + left_cols + right_rows + right_cols


	total_num_type = num_triples + num_bits + num_vectorized
	return total_num_type * constants_dict['m_a_online'] + constants_dict['b_a_online']


def spdz_cost(prog, constants_file):
    # If there is a constants file to do the actual planning
    constants_dict = get_constants_dict(constants_file)    
    d = spdz_offline_cost(prog.req_num, constants_dict)
    print "SPDZ Offline Cost: ", d['cost']
    total_cost = d['cost']
    total_cost += spdz_online_cost(prog.req_num, constants_dict)
    print "SPDZ Online Cost: ", spdz_online_cost(prog.req_num, constants_dict)
    d['total_cost'] = total_cost
    d['decision'] = "SPDZ_" + d['decision']
    return d
       




# --------------- AG-MPC Cost --------------- #
def read_circuit_file(path):
	# Counts the number of AND gates
	num_in = 0
	num_out = 0
	num_ands = 0
	num_gates = 0
	cf = open(path, "r") 
	# First line is the number of gates and wires, so to get the number of gates, we take the first element.
	num_gates = int(cf.readline().split(" ")[0])
	# Second line contains the number of inputs
	inputs = cf.readline().rstrip().split(" ")
	# The last number is the number of output wires. The rest of the numbers are the number of input bits per party.
	num_out = int(inputs.pop(len(inputs) - 1))
	num_in = sum([int(i) for i in inputs])
	# Here count the AND gates in the circuit file.
	for line in cf:
		if "AND" in line:
			num_ands += 1
	cf.close()
	
	return num_in, num_out, num_ands, num_gates

"""
def agmpc_offline(num_in, num_out, num_ands, num_gates, constants_dict):
	num_parties = constants_dict['P']
	total_cost = 0
	# Initialization cost
	total_cost += constants_dict['c_offline_b']
	# Accounting for the number of and triples, both compute and communicaton. f constant is computation, g is communication.
	total_cost += num_ands * (num_parties - 1) * (constants_dict['f1_b'] + constants_dict['g1_b'])
	# Communication cost
	total_cost += (num_parties - 1) * num_gates * constants_dict['g2_b']
	return total_cost
"""
def agmpc_offline(num_in, num_out, num_ands, num_gates, constants_dict):
	# Cost = mx + b, where the x is the number of and gates. They are the main, main overhead.
	num_parties = constants_dict['P']
	offline_cost = 0
	# Initialization cost
	offline_cost += constants_dict['b_b']
	# Accounting for the number of and triples, both compute and communicaton. f constant is computation, g is communication.
	offline_cost += num_ands * (num_parties - 1) * constants_dict['m_b']
	# Communication cost
	#total_cost += (num_parties - 1) * num_gates * constants_dict['g2_b']
	return offline_cost



def agmpc_online(num_gates, constants_dict):
	online_cost = 0
	online_cost += constants_dict['b_b_online'] 
	online_cost += constants_dict['m_b_online'] * num_gates
	return online_cost

def agmpc_cost_helper(num_in, num_out, num_ands, num_gates, constants_dict):
	offline_cost = agmpc_offline(num_in, num_out, num_ands, num_gates, constants_dict)
	online_cost = agmpc_online(num_gates, constants_dict)
	print "AGMPC Offline_cost:", offline_cost
	print "AGMPC Online_cost:", online_cost
	return offline_cost + online_cost

def agmpc_cost(prog, constants_file):
	circuit_file = prog.outfile
	constants_dict = get_constants_dict(constants_file)
	num_in, num_out, num_ands, num_gates = read_circuit_file(circuit_file)
	cost = agmpc_cost_helper(num_in, num_out, num_ands, num_gates, constants_dict)
	d = dict()
	d['decision'] = 'agmpc'
	# All of agmpc's cost is done in ms, so divide the resulting time by 1000.0 to get it in terms of seconds.
	d['total_cost'] = cost
	return d

