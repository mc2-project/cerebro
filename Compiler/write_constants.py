import json

# Common constants
constants = {}

# Number of parties
constants["P"] = 2

# Constants for AG-MPC
# b_b: initialization cost
# m_b: cost for each AND triple
# *_online are the corresponding online costs
constants["b_b"] = 1
constants["m_b"] = 1
constants["b_b_online"] = 1
constants["m_b_online"] = 1

# Constants for SPDZ
# b_a: initialization cost
# m_a: cost for each multiplication triple
# N: SPDZ offline batch size
#
# Linear preprocessing
# m_l: the sub-cost of a single triple that scales linearly with 1/P and N_m
# b: the sub-cost of a single triple that scales linearly with N_m
#
# Quadratic preprocessing
# m_q: the sub-cost of a single triple that scales linearly with P-1 and N_m
# b_q: the sub-cost of a single triple that scales linearly with N_m
#
# Hierarchical preprocessing
# inter_band: inter-region bandwidth
# intra_band: intra-region bandwidth
# c_h, c1_h, c2_h, c3_h, c4_h
#
# SPDZ vectorized preprocessing
# c_v: initialization cost for vectorized triples
# c1_v, c2_v, c3_v
#
# SPDZ online
# m_a_linine: cost that scales with the total number of the different types of triples
# b_a_online: init cost

constants["N"] = 1
constants["m_l"] = 1
constants["b_l"] = 1
constants["m_q"] = 1
constants["b_q"] = 1
constants["inter_band"] = 1
constants["intra_band"] = 1
constants["left_parties"] = 1
constants["right_parties"] = 1
constants["c_h"] = 1
constants["c1_h"] = 1
constants["c2_h"] = 1
constants["c3_h"] = 1
constants["c4_h"] = 1
constants["c_v"] = 1
constants["c1_v"] = 1
constants["c2_v"] = 1
constants["c3_v"] = 1
constants["m_a_online"] = 1
constants["b_a_online"] = 1

f = open("./constants", 'w')
data = json.dumps(constants)
f.write(str(data))
f.close()
