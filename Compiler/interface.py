from types import *
from types_gc import *
import compilerLib, library
import symtable
import re
import numpy as np

SPDZ = 0
GC = 1
LOCAL = 2

class Params(object):
    intp = 64
    f = 32
    k = 64

    @classmethod
    def set_params(cls, int_precision=32, f=32, k=64, parallelism=1):
        cls.intp = int_precision
        cls.f = f
        cls.k = k
        cfix.set_precision(f, k)
        sfix.set_precision(f, k)
        cfix_gc.set_precision(f, k)
        sfix_gc.set_precision(f, k)

class ClearIntegerFactory(object):
    def __call__(self, value):
        if mpc_type == SPDZ:
            return cint(value)
        elif mpc_type == LOCAL:
            return int(value)
        else:
            return cint_gc(Params.intp, value)

class SecretIntegerFactory(object):
    def __call__(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, party)

    def read_input(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, input_party=party)

class SecretIntegerMatrixFactory(object):
    def __call__(self, rows, columns):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")
        if mpc_type == SPDZ:
            ret = sintMatrix(rows, columns)
            return ret
        else:
            ret = sintMatrixGC(rows, columns)
            for i in range(rows):
                for j in range(columns):
                    ret[i][j] = cint_gc(0)
            return ret

class ClearIntegerMatrixFactory(object):
    def __call__(self, rows, columns):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")
        if mpc_type == SPDZ:
            ret = cintMatrix(rows, columns)
            return ret
        else:
            ret = cintMatrixGC(rows, columns)
            for i in range(rows):
                for j in range(columns):
                    ret[i][j] = cint_gc(0)
            return ret

	def read_input(self, rows, columns, channel=0):
		if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")

        if mpc_type == LOCAL:
            raise ValueError("Shouldn't be local.")

        if mpc_type == SPDZ:
            ret = cintMatrix(rows, columns)
            @library.for_range(ret.rows)
            def f(i):
                @library.for_range(ret.columns)
                def g(j):
                    ret[i][j].public_input(channel)
            return ret
        else:
            raise ValueError("Clear matrix read_input not supported for GC")

class ClearFixedPointFactory(object):
    def __call__(self, value):
        if mpc_type == SPDZ:
            return cfix(value)
        elif mpc_type == LOCAL:
            return float(value)
        else:
            return cfix_gc(v=value, scale=True)

class SecretFixedPointFactory(object):
    def read_input(self, party):
        if mpc_type == SPDZ:
            v = sint.get_private_input_from(party)
            vf = sfix.load_sint(v)
            return vf
        else:
            return sfix_gc(v=None, input_party=party)

class SecretFixedPointArrayFactory(object):
    def __call__(self, length):
        if not isinstance(length, int):
            raise ValueError("Array length must be a publicly known integer")
        if mpc_type == SPDZ:
            ret = sfixArray(length)
            return ret
        else:
            ret = sfixArrayGC(length)
            for i in range(length):
                ret[i] = cfix_gc(0)
            return ret

    def read_input(self, length, party):
        if not isinstance(length, int):
            raise ValueError("Array length must be a publicly known integer")
        if mpc_type == SPDZ:
            ret = sfixArray(length)
            @library.for_range(ret.length)
            def f(i):
                v = sint.get_private_input_from(party)
                ret[i] = sfix.load_sint(v, scale=False)
            return ret
        else:
            ret = sfixArrayGC(length)
            for i in range(ret.length):
                ret[i] = sfix_gc(v=None, input_party=party)
            return ret

import struct
class SecretFixedPointMatrixFactory(object):
    def __call__(self, rows, columns):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")
        if mpc_type == LOCAL:
            raise ValueError("Shouldn't be local.")
        if mpc_type == SPDZ:
            ret = sfixMatrix(rows, columns)
            return ret
        else:
            ret = sfixMatrixGC(rows, columns)
            for i in range(rows):
                for j in range(columns):
                    ret[i][j] = cfix_gc(0)
            return ret

    def read_input(self, rows, columns, party):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")

        if mpc_type == LOCAL:
            raise ValueError("Shouldn't be local.")

        if mpc_type == SPDZ:
            ret = sfixMatrix(rows, columns)
            @library.for_range(ret.rows)
            def f(i):
                @library.for_range(ret.columns)
                def g(j):
                    v = sint.get_private_input_from(party)
                    ret[i][j] = sfix.load_sint(v, scale=False)
            return ret
        else:
            ret = sfixMatrixGC(rows, columns)
            for i in range(ret.rows):
                for j in range(ret.columns):
                    ret[i][j] = sfix_gc(v=None, input_party=party)
            return ret


    # Reads input from file.
    def read_clear_input(self, rows, columns, party, f, input_file="../Input_Data/f0"):
        input_type = np.dtype([('f1', np.bool), ('f2', np.int64)])
        lst_inputs = np.fromfile(f, input_type, rows * columns)
        precision = sfix.f
        assert(len(lst_inputs) >= rows * columns)
        res = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                entry = lst_inputs[i * columns + j]
                if entry[0]:
                    factor = -1
                else:
                    factor = 1
                res[i][j] = factor * entry[1] * 1.0 / (2 ** precision)

        return res

    # Read horizontally partitioned data from multiple parties
    # input config should be of the form: (party_id, rows, columns)
    def read_input_variable_rows(self, columns, input_config):
        rows = sum([ic[1] for ic in input_config])
        if mpc_type == SPDZ:
            ret = sfixMatrix(rows, columns)
            party_config = cintMatrix(len(input_config), 2)
            rows_offset = 0
            for (p, r) in input_config:
                @library.for_range(r)
                def a(i):
                    @library.for_range(columns)
                    def b(j):
                        v = sint.get_private_input_from(p)
                        ret[i + rows_offset][j] = sfix.load_sint(v, scale=False)
                rows_offset += r
            return ret
        else:
            ret = sfixMatrixGC(rows, columns)
            rows_offset = 0
            for (p, r) in input_config:
                for i in range(r):
                    for j in range(columns):
                        ret[i+rows_offset][j] = sfix_gc(v=None, input_party=p)
                rows_offset += r
            return ret


class SecretIntegerMatrixFactory(object):
    def __call__(self, rows, columns):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")
        if mpc_type == LOCAL:
            raise ValueError("Shouldn't be local.")
        if mpc_type == SPDZ:
            ret = sintMatrix(rows, columns)
            return ret
        else:
            ret = sintMatrixGC(rows, columns)
            for i in range(rows):
                for j in range(columns):
                    ret[i][j] = cint_gc(0) #sint_gc(Params.intp, party)
            return ret

    def read_input(self, rows, columns, party):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")

        if mpc_type == LOCAL:
            raise ValueError("Shouldn't be local.")

        if mpc_type == SPDZ:
            ret = sintMatrix(rows, columns)
            @library.for_range(ret.rows)
            def f(i):
                @library.for_range(ret.columns)
                def g(j):
                    v = sint.get_private_input_from(party)
                    ret[i][j] = v
            return ret
        else:
            ret = sintMatrixGC(rows, columns)
            for i in range(ret.rows):
                for j in range(ret.columns):
                    ret[i][j] = sint_gc(Params.intp, input_party=party)
            return ret

class ClearFixedPointMatrixFactory(object):
    def __call__(self, rows, columns):
        if mpc_type == SPDZ:
            return cfixMatrix(rows, columns)
        elif mpc_type == LOCAL:
            return np.zeros((rows, columns))
        else:
            ret = cfixMatrixGC(rows, columns)
            for i in range(ret.rows):
                for j in range(ret.columns):
                    ret[i][j] = cfix_gc(0)
            return ret


class PrivateFixedPointMatrix(object):
    def preprocess(self, precision=36):
        input_file="../Input_Data/f0"
        input_type = np.dtype([('f1', np.bool), ('f2', np.int64)])
        lst_inputs = np.fromfile(input_file, input_type)

        data = lst_inputs.flatten().tolist()

        lst_data = []
        for i in range(len(data)):
            entry = data[i]
            if entry[0]:
                factor = -1
            else:
                factor = 1


            val = factor * entry[1] * 1.0 / (2 ** precision)
            lst_data.append(val)

        self.data = lst_data

        #print "READ DATA", self.data

    def read_input(self, rows, columns, party):
        assert(len(self.data) >= rows * columns)
        res = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                entry = self.data.pop(0)
                res[i][j] = entry


        return res


def reveal_all(v, text=""):
    if mpc_type == SPDZ:
        if isinstance(v, (sint, sfix)):
            if text == "":
                text = "value"
            library.print_ln("{} = %s".format(text), v.reveal())
        elif isinstance(v, Array):
            if text == "":
                text = "Array"
            @library.for_range(v.length)
            def f(i):
                library.print_ln("{}[%s] = %s".format(text), i, v[i].reveal())
        elif isinstance(v, Matrix):
            if text == "":
                text = "Matrix"
            @library.for_range(v.rows)
            def f(i):
                @library.for_range(v.columns)
                def g(j):
                    library.print_ln("{}[%s][%s] = %s".format(text), i, j, v[i][j].reveal())
        else:
            raise NotImplemented
    else:
        info = v.reveal(name=text)
        program_gc.output_objects.append(info)


import numpy as np
import struct
# lst_data is a list of matrices right now, sort of hard coded to the specific program
def write_private_data(lst_data):

    lst_private_data = []
    for matrix in lst_data:
        lst_private_data += matrix.flatten().tolist()


    print "PRIVATE DATA OUTPUT: ", lst_private_data
    print "Length: ", len(lst_private_data)
    lst_private_data = [e * pow(2, 36) for e in lst_private_data]
    f = open("../Input_Data" + "/f0", 'w')
    for d in lst_private_data:
        sign = d < 0
        output = struct.pack("?", sign)
        f.write(output)
        output = struct.pack("Q", abs(int(d)))
        f.write(output)
    f.close()




ClearInteger = ClearIntegerFactory()
SecretInteger = SecretIntegerFactory()
ClearFixedPoint = ClearFixedPointFactory()
ClearFixedPointMatrix = ClearFixedPointMatrixFactory()
SecretFixedPoint = SecretFixedPointFactory()
SecretFixedPointArray = SecretFixedPointArrayFactory()
SecretFixedPointMatrix = SecretFixedPointMatrixFactory()
ClearIntegerMatrix = ClearIntegerMatrixFactory()
SecretIntegerMatrix = SecretIntegerMatrixFactory()

compilerLib.VARS["c_int"] = ClearInteger
compilerLib.VARS["s_int"] = SecretInteger
compilerLib.VARS["c_fix"] = ClearFixedPoint
compilerLib.VARS["s_fix"] = SecretFixedPoint
compilerLib.VARS["s_fix_array"] = SecretFixedPointArray
compilerLib.VARS["c_fix_mat"] = ClearFixedPointMatrix
compilerLib.VARS["s_fix_mat"] = SecretFixedPointMatrix
compilerLib.VARS["c_int_mat"] = ClearIntegerMatrix
compilerLib.VARS["s_int_mat"] = SecretIntegerMatrix
compilerLib.VARS["p_mat"] = PrivateFixedPointMatrix
compilerLib.VARS["reveal_all"] = reveal_all
compilerLib.VARS["write_private_data"] = write_private_data


import ast
import astor
from pprint import pprint
import networkx as nx
import operator as op
import astunparse


# Used to convert expressions with binary operators like 5 * 6 into values.
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg, ast.Eq: op.eq, ast.Lt: op.lt, ast.Gt: op.gt, ast.GtE: op.ge, ast.LtE: op.le}

class ForLoopParser(ast.NodeTransformer):
    def __init__(self):
        self.forloop_counter = 0

    def parse_for(self, node):
        dec = ast.Call(func=ast.Name(id="for_range", ctx=ast.Load()), args=node.iter.args, keywords=[])
        new_node = ast.FunctionDef(name="for{}".format(self.forloop_counter),
                                   args=ast.arguments(args=[node.target], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                                   body=node.body,
                                   keywords=[],
                                   decorator_list=[dec])

        self.forloop_counter += 1
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node

    def visit_For(self, node):
        self.generic_visit(node)

        if isinstance(node.iter, ast.Call):
            if node.iter.func.id == "range":
                if mpc_type == SPDZ:
                    return self.parse_for(node)
                else:
                    return node

        raise ValueError("For loop only supports style 'for i in range(...)'")



class ProcessDependencies(ast.NodeVisitor):
    def __init__(self, target_fn="matmul"):
        # Special cases for function calls to the mllib that used matmul calls.
        self.mllib_fn = ["LogisticRegression", "ADMM", "ADMM_preprocess"]
        # This helps with tracking which code is defined but never actually used/called in the .mpc file, so don't overcount.
        self.global_scope = "GLOBAL"
        # map function name to function definition ast object.
        self.functions = {}
        self.scope_stack = []
        # If encounter target function call, instantly add 1
        # Functions are nodes, dependencies are edges
        self.G = nx.DiGraph()
        # Function we want to keep track of, currently it's "matmul", but this variable is here in case we want to extend it to count other fns.
        self.target_fn = target_fn
        # Add a node so that if a function is called, draw an edge from
        self.G.add_node(self.global_scope)


        self.fns_to_params = {}

    def visit_FunctionDef(self, node):
        # Not considering differing scopes and same name.
        if node.name != self.target_fn and node.name not in self.mllib_fn:
            self.functions[node.name] = node
            #print "Private type inference, visit function def: ", node.name
            args = []
            for arg in node.args.args:
                args.append(arg.id)

            self.fns_to_params[node.name] = args
        else:
            new_target_fn_name = node.name + str(node.lineno)
            self.G.add_node(new_target_fn_name)
            self.functions[new_target_fn_name] = node

        #print "Visiting function: ", node.name
        if len(self.scope_stack):
            "Function ", node.name, " was called by ", self.scope_stack[0]

        # Enter a function, put it on the stack
        self.scope_stack.insert(0, node.name)
        # Visit the children in the function
        self.generic_visit(node)
        self.scope_stack.pop(0)


    def visit_Call(self, node):
        parent = self.get_parent()
        # Some nodes don't have ids?
        if 'id' in node.func.__dict__.keys():
            fn_name = node.func.id
            # Add in extra identifier.
            new_target_fn_name = fn_name+str(node.lineno)
            #print "Function {0} called by {1}".format(fn_name, parent)
            self.scope_stack.insert(0, fn_name)
            #print "First time visiting: ", fn_name
            self.generic_visit(node)


            self.process_fn_call(parent, fn_name, new_target_fn_name, node.lineno)




            self.scope_stack.pop(0)


    # new_target_fn_name is used to differentiate multiple calls to the same library function such as matmul and admm.
    # Currently importing other files does not work.
    def process_fn_call(self, parent, fn_name, new_target_fn_name, lineno):
        if fn_name == self.target_fn or fn_name in self.mllib_fn:
            self.G.add_node(new_target_fn_name)
            self.G.add_edge(parent, new_target_fn_name)
        else:
            self.G.add_edge(parent, fn_name)



        if fn_name in self.mllib_fn:
            # For each mllib call, have to add extra nodes to indicate matmul calls.
            if fn_name == "LogisticRegression":
                # Logistic Regression calls matmul on 2 separate occasions, so add fake nodes in the call graph to reflect this.
                matmul_node1 = "matmulLR1_" + str(lineno)
                matmul_node2 = "matmulLR2_" + str(lineno)
                self.G.add_node(matmul_node1)
                self.G.add_node(matmul_node2)
                self.G.add_edge(new_target_fn_name, matmul_node1)
                self.G.add_edge(new_target_fn_name, matmul_node2)
            elif fn_name == "ADMM":
                # matmul used for admm_local function
                matmul_node1 = "matmulADMM1_" + str(lineno)
                self.G.add_node(matmul_node1)
                self.G.add_edge(new_target_fn_name, matmul_node1)
            elif fn_name == "ADMM_preprocess":
                # matmul used for X_i^T X_i
                matmul_node2 = "matmulADMM_preprocess1_" + str(lineno)
                # matmul used for X_i^T y_i
                matmul_node1 = "matmulADMM_preprocess2_" + str(lineno)
                self.G.add_node(matmul_node1)
                self.G.add_node(matmul_node2)
                self.G.add_edge(new_target_fn_name, matmul_node1)
                self.G.add_edge(new_target_fn_name, matmul_node2)


    def visit_For(self, node):
        # Parse the for loop to get the # of iterations. Won't work if variable # of iterations so try to avoid that.
        #print "For loop lineno: ", node.lineno
        self.functions[node.lineno] = node

        #print "Parent of for: ", self.get_parent()

        if node.lineno not in self.G.nodes:
            self.G.add_node(node.lineno)


        parent = self.get_parent()
        #print "For loop: Adding edge from: {0} to {1}".format(parent, node.lineno)
        self.G.add_edge(parent, node.lineno)
        self.scope_stack.insert(0, node.lineno)
        #print "Visiting for loop in function: ", self.scope_stack[0]
        for item in node.body:
            self.visit(item)

        self.scope_stack.pop(0)

    def visit_Assign(self, node):
        self.visit(node.value)


    def get_parent(self):
        if len(self.scope_stack):
            return self.scope_stack[0]
        else:
            return self.global_scope


class CountFnCall(ast.NodeVisitor):

    def __init__(self, G, functions, target_fn="matmul"):
        # List of function_calls in mllib
        # Should treat these matmul calls as SUPER Matmul calls.
        # Special function calls to mllib that use matmul
        self.mllib_fn = ["LogisticRegression", "ADMM", "ADMM_preprocess"]
        self.global_scope = "GLOBAL"
        # Maps function names to function_def nodes.
        self.functions = functions
        # Maps function name to how many time that block calls matmul
        self.fns_to_calls = {}
        # Maps each matmul to the dimensions of the matmul.
        self.matmul_to_dims = {}
        # Maps a matmul node to how many times it has been called.
        self.matmul_to_calls = {}
        self.G = G.copy()
        self.target_fn = target_fn
        self.counter = 0
        # Maps for loop to the number of iterations.
        self.for_to_iters = {}
        # List of different matmul nodes. A node represents a unique lineno in which matmul is invoked
        self.lst_matmuls = []
        self.scope_stack = []
        # Maps dimension of matmul (which is a tuple) to the # of times that matmul is invoked.
        self.vectorized_calls = {}
        self.preprocess()
        # Gather all the necessary information for counting calls.
        self.process()


    # Add in all the matmul nodes. Set each 'matmul' function in fns_to_calls to 1 since each matmul calls matmul once.
    def preprocess(self):
        for fn_name in self.G.nodes():
            #print "Function name: ", fn_name
            if self.target_fn in str(fn_name):
                self.fns_to_calls[fn_name] = 1
                self.matmul_to_calls[fn_name] = 0
                self.lst_matmuls.append(fn_name)


        #print "List of matmul calls: ", self.lst_matmuls

    def process(self):
        topological_ordering = list(reversed(list(nx.topological_sort(self.G))))
        #print "Topological ordering: ", topological_ordering
        for node in topological_ordering:
            # Has to actually be called
            if node in self.functions.keys() and len(self.G.in_edges(node)):
                self.visit(self.functions[node])

        # Reset the counter
        self.counter = 0


    def postprocess(self):
        # Remove function_calls that never get used
        lst_pop_fns = []
        for fn_name in self.fns_to_calls.keys():
            if fn_name in self.G.nodes() and not len(self.G.in_edges(fn_name)) and fn_name != self.global_scope:
                lst_pop_fns.append(fn_name)


        # Iteratively remove any nodes that never get reached, or with 0 in edges.
        while len(lst_pop_fns):
            for fn_name in lst_pop_fns:
                #print "Popped function: ", fn_name
                self.G.remove_node(fn_name)
            # Popped all the function, reset list
            lst_pop_fns = []

            for fn in self.G.nodes():
                if not len(self.G.in_edges(fn)) and fn != self.global_scope:
                    lst_pop_fns.append(fn)


        for node in list(nx.topological_sort(self.G)):
            for neighbor in self.G.neighbors(node):
                # We encountered a for-loop.
                if node in self.for_to_iters.keys():
                    num_iter = self.for_to_iters[node]
                    self.G[node][neighbor]['weight'] = num_iter
                else:
                    # Look at node's parent for for-loops.
                    multiplicative_factor = self.propagate_for(node)
                    print "Count fn_call Postprocess: {0}, {1}".format(node, neighbor)
                    self.G[node][neighbor]['weight'] = self.fns_to_calls.get(neighbor, 0) * multiplicative_factor

                if self.G[node][neighbor]['weight'] != 0:
                    print "Count fn_call WEIGHT", node, neighbor, self.G[node][neighbor]['weight']


        # Extract info from the edges. # of calls to a matmul is the sum of the incoming weights to each matmul node.
        for matmul in self.lst_matmuls:
            # May have been popped earlier
            if matmul in self.G.nodes():
                num_matmul_calls = 0
                for u, v, data in self.G.in_edges(matmul, data=True):
                    num_matmul_calls += data['weight']

                self.matmul_to_calls[matmul] += num_matmul_calls


                # Finally, combine matmul nodes that have the same dimension
                dims = self.matmul_to_dims[matmul]
                self.process_mat_dims(matmul, dims)


    def process_mat_dims(self, matmul, dim):
        if tuple(dim) not in self.vectorized_calls.keys():
            self.vectorized_calls[tuple(dim)] = self.matmul_to_calls[matmul]
        else:
            self.vectorized_calls[tuple(dim)] += self.matmul_to_calls[matmul]


    def propagate_for(self, u):
        multiplicative_factor = 1
        child = u
        for parent, _ in self.G.in_edges(u):
            # KEY ASSUMPTION THAT CANNOT HAVE > 1 direct parent that is a for loop. Doesn't make any sense.
            if parent in self.for_to_iters.keys():
                while parent in self.for_to_iters.keys():
                    # The edges encode how many times the parent calls the child.
                    multiplicative_factor *= self.G[parent][child]['weight']
                    found_ancestor = False
                    for grandparent, _ in self.G.in_edges(parent):
                        if grandparent in self.for_to_iters.keys():
                            child = parent
                            parent = grandparent
                            found_ancestor = True
                            break

                    if not found_ancestor:
                        break

                break

        return multiplicative_factor

    def visit_FunctionDef(self, node):
        #print "Visit function: ", node.name
        before_visit = self.counter
        self.scope_stack.insert(0, node.name)
        self.generic_visit(node)
        self.scope_stack.pop(0)
        after_visit = self.counter
        diff = after_visit - before_visit
        if node.name not in self.fns_to_calls.keys():
            self.fns_to_calls[node.name] = diff

        self.counter = before_visit

    def visit_For(self, node):
        try:
            if len(node.iter.args) == 1:
                # Ex: for i in range(6)
                num_iter = self.eval_args_helper(node.iter.args[0])
            elif len(node.iter.args) == 2:
                # Start and an end
                # Ex: for i in range(6, 8)
                num_iter = self.eval_args_helper(node.iter.args[1]) - self.eval_args_helper(node.iter.args[0])
            else:
                # Start, end and a step
                # Ex: for i in range(6, 12, 2)
                num_iter = (self.eval_args_helper(node.iter.args[1]) - self.eval_args_helper(node.iter.args[0])) / self.eval_args_helper(node.iter.args[2])
        except Exception as e:
            return


        self.for_to_iters[node.lineno] = num_iter
        if node.lineno in self.fns_to_calls.keys():
            self.counter += self.fns_to_calls[node.lineno]
            return

        before_visit = self.counter
        #print "In for loop: ", node.lineno
        for item in node.body:
            if isinstance(item, ast.For) and item.lineno in self.fns_to_calls.keys():
                self.counter += self.fns_to_calls[item.lineno]
            else:
                self.scope_stack.insert(0, node.lineno)
                self.visit(item)
                self.scope_stack.pop(0)

        after_visit = self.counter
        diff = after_visit - before_visit
        if node.lineno not in self.fns_to_calls.keys():
            self.fns_to_calls[node.lineno] = diff * num_iter

        #print "For loop {0} calls the target function {1} times".format(node.lineno, diff * num_iter)


    def eval_args_helper(self, node):
        if hasattr(node, 'n'):
            return node.n
        else:
            left_val = self.eval_args_helper(node.left)
            right_val = self.eval_args_helper(node.right)
            res = operators[type(node.op)](left_val, right_val)
            return res

    def visit_Call(self, node):
        before_visit = self.counter
        # An actual function call like f() rather than a method like a.f()
        if 'id' in node.func.__dict__.keys():
            fn_name = node.func.id
            #print "Calling function: ", fn_name
            if fn_name not in self.fns_to_calls.keys():
                self.scope_stack.insert(0, fn_name)
                self.generic_visit(node)
                self.scope_stack.pop(0)
                after_visit = self.counter
                diff = after_visit - before_visit
                self.fns_to_calls[fn_name] = diff

            if fn_name == self.target_fn:
                new_target_fn_name = fn_name + str(node.lineno)
                # Contains left_mat rows, cols, then right_mat rows, cols
                lst_dims = []
                # Matmul dimension arguments are arg # 2, 3, 4, 5
                for i in range(2, 6):
                    # Means it's a constant value
                    val = self.eval_args_helper(node.args[i])
                    lst_dims.append(val)

                # Matmul dimension argument #7 is the type of the matrices.
                lst_dims.append(node.args[6].id)

                self.matmul_to_dims[new_target_fn_name] = tuple(lst_dims)
                self.counter = before_visit + self.fns_to_calls[new_target_fn_name]

                # A bare matmul call not in the enclosing scope of any other function.
                #if not len(self.scope_stack):
                    #self.matmul_to_calls[new_target_fn_name] += 1
            elif fn_name in self.mllib_fn:
                new_target_fn_name = fn_name + str(node.lineno)
                if fn_name == "LogisticRegression":
                    lst_info = []
                    # VERY HARDCODED
                    # lst_info will contain batch_size, sgd_iters, dimension of data.
                    for i in range(2, 5):
                        # Means it's a constant value
                        # Evaluate the
                        val = self.eval_args_helper(node.args[i])
                        lst_info.append(val)

                    print "LR info: "
                    batch_size = lst_info[0]
                    sgd_iters = lst_info[1]
                    dim = lst_info[2]

                    # Assume working with fixed point. Saves an extra parameter ugh.
                    dim1 = (batch_size, dim, dim, 1, 'sfix')
                    dim2 = (dim, batch_size, batch_size, 1, 'sfix')

                    LR_matmul_node1 = "matmulLR1_" + str(node.lineno)
                    LR_matmul_node2 = "matmulLR2_" + str(node.lineno)
                    # UGH, this is such bad code. This is for calling library functions that don't do the same # of matmuls for each dimensions.
                    self.fns_to_calls[new_target_fn_name] = 2 * sgd_iters
                    self.fns_to_calls[LR_matmul_node1] = sgd_iters
                    self.fns_to_calls[LR_matmul_node2] = sgd_iters
                    self.matmul_to_dims[LR_matmul_node1] = dim1
                    self.matmul_to_dims[LR_matmul_node2] = dim2

                elif fn_name == "ADMM":
                    lst_info = []
                    for i in range(2, 5):
                        val = self.eval_args_helper(node.args[i])
                        lst_info.append(val)

                    admm_iter = lst_info[0]
                    num_parties = lst_info[1]
                    num_cols = lst_info[2]
                    ADMM_matmul_node1 = "matmulADMM1_" + str(node.lineno)
                    matmul_dim = (num_cols, num_cols, num_cols, 1, 'sfix')
                    self.fns_to_calls[new_target_fn_name] = admm_iter * num_parties
                    self.fns_to_calls[ADMM_matmul_node1] = admm_iter * num_parties
                    self.matmul_to_dims[ADMM_matmul_node1] = matmul_dim

                elif fn_name == "ADMM_preprocess":
                    lst_info = []
                    for i in range(3, 6):
                        val = self.eval_args_helper(node.args[i])
                        lst_info.append(val)

                    num_parties = lst_info[0]
                    num_rows = lst_info[1]
                    num_cols = lst_info[2]
                    ADMM_preprocess_matmul_node1 = "matmulADMM_preprocess1_" + str(node.lineno)
                    ADMM_preprocess_matmul_node2 = "matmulADMM_preprocess2_" + str(node.lineno)

                    # Dimensions for X_i^T X_i
                    matmul_dim1 = (num_cols, num_rows, num_rows, num_cols, 'sfix')
                    # Dimensions for X_i^T y_i
                    matmul_dim2 = (num_cols, num_rows, num_rows, 1, 'sfix')

                    self.fns_to_calls[new_target_fn_name] = 2 * num_parties
                    self.fns_to_calls[ADMM_preprocess_matmul_node1] = num_parties
                    self.fns_to_calls[ADMM_preprocess_matmul_node2] = num_parties
                    self.matmul_to_dims[ADMM_preprocess_matmul_node1] = matmul_dim1
                    self.matmul_to_dims[ADMM_preprocess_matmul_node2] = matmul_dim2

            else:
                self.counter = before_visit + self.fns_to_calls[fn_name]




    def visit_Assign(self, node):
        self.visit(node.value)


import copy
# TODO: Incorporate scope possibly.
class ConstantPropagation(ast.NodeTransformer):
    """NodeTransformer that will inline any Number and String
    constants defined on the module level wherever they are used
    throughout the module. Any Number or String variable matching [A-Z_]+
    on the module level will be used as a constant"""

    def __init__(self):
        self._constants = {}

    def visit_Module(self, node):
        """Find eglible variables to be inlined and store
        the Name->value mapping in self._constants for later use"""
        assigns = [x for x in node.body if type(x) == ast.Assign]
        for assign in assigns:
            if type(assign.value) in (ast.Num, ast.Str):
                for name in assign.targets:
                    self._constants[name.id] = assign.value
        return self.generic_visit(node)

    def visit_Name(self, node):
        """If node.id is in self._constants, replace the
        loading of the node with the actual value"""
        for k in self._constants.keys():
            if node.id == k:
                #print "Name: {0}, value: {1}".format(node.id, self._constants[k].n)
                #print node.__dict__
                return ast.Num(self._constants[k].n) #self._constants[k]

        return node

    def visit_Assign(self, node):
        copy_assign = copy.deepcopy(node) #ast.Assign(value=node.value, targets=node.targets)
        #node.value = self.visit(node.value)
        new_assign_val = self.visit(copy_assign.value)
        if new_assign_val:
            copy_assign.value = new_assign_val
        try:
            # No multiassignment such as a,b = c,d
            if not isinstance(copy_assign.value, ast.Tuple):
                if type(copy_assign.value) not in (ast.Call, ast.Name):
                    try:
                        val = self.eval_args_helper(copy_assign.value)
                        copy_assign.value = ast.Num(n=val)
                        self._constants[copy_assign.targets[0].id] = copy_assign.value
                    except:
                        self._constants.pop(copy_assign.targets[0].id)

                # So far don't allow multi-assignment, not sure how to go about this.

            else:
                for i in range(len(copy_assign.value.elts)):
                    try:
                        obj = copy_assign.value.elts[i]
                        if type(obj) not in (ast.Call, ast.Name):
                            val = self.eval_args_helper(obj)
                            self._constants[copy_assign.targets[0].elts[i].id] = val
                            obj.value = val
                    except Exception as e:
                        print "Exception in assign multiassignment: ", e
                        self._constants.pop(copy_assign.targets[0].elts[i].id)

                #print "Multiassignment not supported: ", node.value.__dict__
        except Exception as e:
            # For some reason, cannot evaluate the right hand side
            # print e


        # Visit the left hand side of the assignment

        if isinstance(copy_assign.targets[0], ast.Subscript):
            copy_assign.targets[0] = self.visit(copy_assign.targets[0])

        return copy_assign




    def visit_Subscript(self, node):
        return self.generic_visit(node)

    def eval_args_helper(self, node):
        if hasattr(node, 'n'):
            return node.n
        else:
            left_val = self.eval_args_helper(node.left)
            right_val = self.eval_args_helper(node.right)
            res = operators[type(node.op)](left_val, right_val)
            return res

    def visit_BinOp(self, node):
        try:
            val = self.eval_args_helper(node)
            return ast.Num(n=val)
        except Exception as e:
            #print e
            #print "ConstantPropagation Exception"
            return self.generic_visit(node)


    def visit_Compare(self, node):
        rename_left = self.visit(node.left)
        if rename_left:
            node.left = rename_left
        for i in range(len(node.comparators)):
            rename_comparator = self.visit(node.comparators[i])
            if rename_comparator:
                node.comparators[i] = rename_comparator

        if len(node.ops) > 1:
            print "Doesn't support more than 1 comparison operator at a time."
            return node

        try:
            val = self.eval_compare_helper(node)
            return val

        except Exception as e:
            #print e
            return node


    def eval_compare_helper(self, node):
        left = node.left
        comapre_op = node.ops[0]
        compare = node.comparators[0]
        # Evaluate the compare operator against the two arguments. Currently only support single comparison statements.
        if isinstance(left, ast.Num) and isinstance(compare, ast.Num):
            compare_res = operators[type(node.ops[0])](left.n, compare.n)
            return ast.Num(n=compare_res)
        else:
            return None


class ASTChecks(ast.NodeTransformer):
    def __init__(self):
        self.if_stack = []
        self.test_name = "test"
        self.test_counter = 0
        self.scope_level = 0
        self.assignments = {} # indexed by the assignment targets, along with the assigned values
        self.sub_assignments = {}
        self.depth = 0

    def negate_condition(self, cond):
        return ast.BinOp(left=ast.Num(1), op=ast.Sub(), right=cond)

    def merge_test_single(self, test):
        if test[1]:
            return ast.Name(id=test[0], ctx=ast.Load())
        else:
            return self.negate_condition(ast.Name(id=test[0], ctx=ast.Load()))

    # Given a list of (condition_name, True/False), merge these conditions into a single condition
    def merge_test_list(self, test_list):
        ret = test_list[0]
        ret = self.merge_test_single(ret)
        for test in test_list[1:]:
            ret = ast.BinOp(left=ret, op=ast.Mult(), right=self.merge_test_single(test))

        return ret

    # First, merge the conditions into a single condition, then negate the entire condition
    def neg_merge_test_list(self, test_list):
        test_v = self.merge_test_list(test_list)
        return self.negate_condition(test_v)

    def assign_transform(self, target, test_list, a, b):
        test_v = self.merge_test_list(test_list)
        left_sum = ast.BinOp(left=test_v, op=ast.Mult(), right=a)
        test_neg = ast.BinOp(left=ast.Num(1), op=ast.Sub(), right=test_v)
        right_sum = ast.BinOp(left=test_neg, op=ast.Mult(), right=b)
        return ast.Assign(targets=[target], value=ast.BinOp(left=left_sum, op=ast.Add(), right=right_sum))

    # Given multiple conditions of the form (condition_list, value)
    # where condition_list is a list of conditions, and value is a RHS single value for assignment
    def assign_transform_multi(self, target, conditions_list):
        current_sum = None
        current_condition_list = []
        for (test_list, value) in conditions_list:
            test_v = self.merge_test_list(test_list)
            current_condition_list += test_list
            prod = ast.BinOp(left=test_v, op=ast.Mult(), right=value)
            if current_sum is None:
                current_sum = prod
            else:
                current_sum = ast.BinOp(left=current_sum, op=ast.Add(), right=prod)

        neg_cond = self.neg_merge_test_list(current_condition_list)
        final_prod = ast.BinOp(left=neg_cond, op=ast.Mult(), right=target)
        current_sum = ast.BinOp(left=current_sum, op=ast.Add(), right=final_prod)

        return ast.Assign(targets=[target], value=current_sum)

    def visit_If(self, node):
        if not isinstance(node.test, ast.Compare):
            raise ValueError("Currently, the if conditional has to be a single Compare expression")
        if len(node.body) > 1:
            raise ValueError("We also don't allow multiple statements inside an if statement")

        statements = []
        test_name = self.test_name + str(self.test_counter)
        self.test_counter += 1
        test_assign = ast.Assign(targets=[ast.Name(id=test_name, ctx=ast.Store())], value=node.test)
        statements.append(test_assign)

        self.if_stack.append((test_name, True))
        self.depth += 1
        for n in node.body:
            if isinstance(n, ast.If):
                statements += self.visit(n)
            elif isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in self.assignments:
                            self.assignments[target.id] = []
                        self.assignments[target.id].append(([x for x in self.if_stack], n.value))
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        if target.value.id not in self.sub_assignments:
                            self.sub_assignments[target.value.id] = [[], []]
                        self.sub_assignments[target.value.id][0].append(([x for x in self.if_stack], target.slice))
                        self.sub_assignments[target.value.id][1].append(([x for x in self.if_stack], n.value))
            else:
                raise ValueError("Does not support non-assignment statements within if statements")
        self.depth -= 1
        self.if_stack.pop()

        self.if_stack.append((test_name, False))
        self.depth += 1
        for n in node.orelse:
            if isinstance(n, ast.If):
                statements += self.visit(n)
            elif isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in self.assignments:
                            self.assignments[target.id] = []
                        self.assignments[target.id].append(([x for x in self.if_stack], n.value))
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        if target.value.id not in self.sub_assignments:
                            self.sub_assignments[target.value.id] = [[], []]
                        self.sub_assignments[target.value.id][0].append(([x for x in self.if_stack], target.slice))
                        self.sub_assignments[target.value.id][1].append(([x for x in self.if_stack], n.value))
            else:
                raise ValueError("Does not support non-assignment statements within if statements")
        self.depth -= 1
        self.if_stack.pop()

        if self.depth == 0:
            for (name, test_list) in self.assignments.iteritems():
                statement = self.assign_transform_multi(ast.Name(id=name, ctx=ast.Store), test_list)
                statements.append(statement)
            for (name, l) in self.sub_assignments.iteritems():
                statements.append(ast.Assign(targets=[ast.Name(id=name+"_index", ctx=ast.Load)], value=ast.Num(-1)))
                statements.append(ast.Assign(targets=[ast.Name(id=name+"_value", ctx=ast.Load)], value=ast.Num(-1)))

                index_test_list = l[0]
                index_statement = self.assign_transform_multi(ast.Name(id=name+"_index", ctx=ast.Store), index_test_list)
                value_test_list = l[1]
                value_statement = self.assign_transform_multi(ast.Name(id=name+"_value", ctx=ast.Store), value_test_list)
                statements.append(index_statement)
                statements.append(value_statement)
                sub = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load), slice=ast.Name(id=name+"_index", ctx=ast.Load), ctx=ast.Store)
                assign = ast.Assign(targets=[sub], value=ast.Name(id=name+"_value", ctx=ast.Load))
                statements.append(assign)

        ast.copy_location(statements[0], node)
        counter = 0
        for s in statements:
            s.lineno = statements[0].lineno + counter
            s.col_offset = statements[0].col_offset
            counter += 1

        return statements





from enum import Enum
import functools

class MC2_Types(Enum):
    SECRET = "SECRET"
    CLEAR = "CLEAR"
    PRIVATE = "PRIVATE"


import heapq
from ordered_set import OrderedSet
# Gather information about the program

# Combine dimension inference here

class ProgramSplitterHelper(ast.NodeVisitor):

    def __init__(self):
        # Hardcoded for SCALE-MAMBA
        self.secret_types = ["sint", "sfix", "s_fix", "s_fix_mat", "sfixMatrix", "sintMatrix", "Piecewise"]
        self.clear_types = ["cint", "cfix", "c_fix", "cfixMatrix", "c_fix_mat"]
        self.python_clear = ["True", "False"]
        self.private_inputs = ["read_input_from", "read_input"]
        self.aggregation_fns = ["sum", "avg", "min_lib", "max_lib", "sum_lib", "reduce_lib"]
        self.reduce_fn = ["reduce_lib"]
        self.lst_private_vars = []
        self.var_graph = nx.DiGraph()
        # Key is (name, lineno), value is the type of the variable
        self.var_tracker = {}

        # Map (mat_name, lineno) to dimension of matrix
        self.mat_to_dim = {}
        self.vectorized_calls = {}

        self.name_to_node = {}



        # Maps variable name to its counter.
        self.var_to_count = {}
        # A global counter to keep track of relative ordering of variables.
        self.global_counter = 0
        self.pq = []


        # Map lists (special types) to the list of types, vars they contain
        self.lst_to_types = {}
        self.aggregations = {}

    # Skip function definitions. Assume entire program has been unrolled.
    def visit_FunctionDef(self, node):
        return


    # Function calls like .append(), needs to be marked secret
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            try:
            #print "ATTRIBUTE HELP:", node.__dict__, node.func.__dict__
                name = node.func.value.id
            except Exception as e:
                #print astunparse.unparse(node)
                #print "The above line is causing error. Pls Fix!"
                return
            if self.name_exists(name):
                next_lineno = self.get_next_var_num(name)
                self.var_graph.add_node((name,next_lineno), mc2_type=self.lookup_type(name), op=node)
                self.name_to_node[(name, next_lineno)] = node

                args = node.args
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        if self.name_exists(arg.id):
                            self.var_graph.add_edge((arg.id, self.var_to_count[arg.id]), (name, next_lineno))

            # Deal with this dimension crap.
            try:
                ref_to_name = self.lookup(name)
            except ValueError as e:
                return

            attr = node.func.attr
            first_arg = node.args[0].id
            if ref_to_name in self.mat_to_dim.keys():
                if attr in ("append", "placeholder"):
                    ref_to_first_arg = self.lookup(first_arg)
                    if ref_to_first_arg in self.mat_to_dim.keys():
                        self.mat_to_dim[ref_to_name].append(self.mat_to_dim[ref_to_first_arg])
                    else:
                        self.mat_to_dim[ref_to_name].append(None)


                mc2_type = self.var_tracker[ref_to_first_arg]
                self.lst_to_types[ref_to_name].append((ref_to_first_arg, mc2_type))



    def add_to_graph(self, node, var_name, mc2_type):
        lineno = self.get_next_var_num(var_name)
        self.var_tracker[(var_name, lineno)] = mc2_type
        self.var_graph.add_node((var_name, lineno), mc2_type=mc2_type, op=node, counter=lineno)
        self.name_to_node[(var_name, lineno)] = node


    def process_mat_declare(self, name, lineno, args):
        rows = args[0].n
        cols = args[1].n
        self.mat_to_dim[(name, lineno)] = (rows, cols)



    # HOLY SHIT this is very messy.
    def visit_Assign(self, node):
        # multi-assignment, figure out later
        if isinstance(node.targets[0], ast.Tuple):
            lst_var_names = []
            for name_obj in node.targets[0].elts:
                lst_var_names.append(name_obj.id)

            # Assume right now don't have library functions that return multiple values.
            if isinstance(node.value, ast.Call):
                fn_name = self.get_fn_name(node)
                if fn_name in self.secret_types:
                    for var_name in lst_var_names:
                        self.add_to_graph(node, var_name, (MC2_Types.SECRET, ""))
                # a = cfix(5)
                elif fn_name in self.clear_types:
                    for var_name in lst_var_names:
                        self.add_to_graph(node, var_name, (MC2_Types.CLEAR, ""))

                # HACKY Way just in case another case didn't consider, basically this is calling library functions.
                else:
                    if len(node.value.args) > 1:
                        res_type = self.check_types(node.value.args)
                    else:
                        if isinstance(node.value.args[0], ast.Num):
                            res_type = (MC2_Types.CLEAR, "")
                        else:
                            res_type = self.lookup_type(node.value.args[0].id)


                    for var_name in lst_var_names:
                        new_lineno = self.get_next_var_num(var_name)
                        self.var_tracker[(var_name, new_lineno)] = res_type

                    # Track dependencies between input and output
                    self.track_input_output_dependencies(node.value.args, lst_var_names, node, res_type)

            # Multiassignment of variables like a,b = c, d, currently barely supported.
            # Probably do not want to support this. The graph gets completely fucked up by this.
            elif isinstance(node.value, ast.Tuple):
                raise ValueError("No multiassignment allowed for now, until I figure out a way to fix this")
            """
            elif isinstance(node.value, ast.Tuple):
                print "MULTIASSIGNMENT"
                right_hand_side_var_names = []
                for name_obj in node.value.elts:
                    right_hand_side_var_names.append(name_obj.id)

                # Make sure number of vars on LHS == number of vars on RHS
                assert(len(lst_var_names) == len(right_hand_side_var_names))
                next_lineno = self.get_next_var_num(lst_var_names[0])
                for lhs_var_name in lst_var_names:
                    self.var_to_count[lhs_var_name] = next_lineno

                mc2_type = self.check_types([ast.Name(id=rhs_var_name) for rhs_var_name in right_hand_side_var_names])
                self.var_graph.add_node((tuple(lst_var_names), next_lineno), op=node, mc2_type=mc2_type)
                for lhs_var_name in lst_var_names:
                    self.var_tracker[(lhs_var_name, self.var_to_count[lhs_var_name])] = mc2_type
                for i in range(len(right_hand_side_var_names)):
                    lhs_var_name = lst_var_names[i]
                    rhs_var_name = right_hand_side_var_names[i]
                    most_recent_right_ref = self.lookup(rhs_var_name)
                    if most_recent_right_ref in self.mat_to_dim.keys():
                        self.mat_to_dim[(lhs_var_name, next_lineno)] = self.mat_to_dim[most_recent_right_ref]

                    self.var_graph.add_edge((rhs_var_name, self.var_to_count[rhs_var_name]), (tuple(lst_var_names), self.var_to_count[lhs_var_name]))

                for i in range(len(lst_var_names)):
                    lhs_var_name = lst_var_names[i]
                    rhs_var_name = right_hand_side_var_names[i]
                    mc2_type = self.check_(rhs_var_name)
                    most_recent_right_ref = self.lookup(rhs_var_name)
                    #self.add_to_graph(node, lhs_var_name, mc2_type)
                    self.var_graph.add_edge((rhs_var_name, self.var_to_count[rhs_var_name]), (lhs_var_name, self.var_to_count[lhs_var_name]))
                    next_lineno = self.var_to_count[lhs_var_name]
                    if most_recent_right_ref in self.mat_to_dim.keys():
                        self.mat_to_dim[(lhs_var_name, next_lineno)] = self.mat_to_dim[most_recent_right_ref]
            """



        else:
            # Encounter subscript object like A[i] = b
            if isinstance(node.targets[0], ast.Subscript):
                subscript_obj = node.targets[0]
                subscript_name = self.get_subscript_name(subscript_obj)
                next_lineno = self.get_next_var_num(subscript_name)
                self.var_graph.add_node((subscript_name, next_lineno), mc2_type=self.lookup_type(subscript_name), op=node)
                self.name_to_node[(subscript_name, next_lineno)] = node

            # Something like a = 4
            elif isinstance(node.value, ast.Num):
                self.add_to_graph(node, node.targets[0].id, (MC2_Types.CLEAR, ""))


            elif isinstance(node.value, ast.Compare):
                # Assume single comparators
                mc2_type = self.check_types(node.value.comparators + [node.value.left])


                self.add_to_graph(node, node.targets[0].id, mc2_type)
            # Calling .read_input essentially
            elif self.check_private_input(node):
                if not isinstance(node.value.args[len(node.value.args) - 1], ast.Num):
                    raise ValueError("Last argument of read_input is a party which must be a number!")

                party_num = node.value.args[len(node.value.args) - 1].n
                self.add_to_graph(node, node.targets[0].id, (MC2_Types.PRIVATE, party_num))
                self.process_mat_declare(node.targets[0].id, self.var_to_count[node.targets[0].id], node.value.args)

            # Calling a function that is a library function
            elif isinstance(node.value, ast.Call):

                # a = sfix(5)
                fn_name = self.get_fn_name(node.value)
                if fn_name in self.secret_types:
                    self.add_to_graph(node, node.targets[0].id, (MC2_Types.SECRET, ""))
                    # Add dimension
                    if fn_name in ("s_fix_mat", "sfixMatrix"):
                        self.process_mat_declare(node.targets[0].id, self.var_to_count[node.targets[0].id], node.value.args)


                # a = cfix(5)
                elif fn_name in self.clear_types:
                    #self.var_tracker[(node.targets[0].id, next_lineno)] = (MC2_Types.CLEAR, "")
                    #self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=(MC2_Types.CLEAR, ""), op=node)
                    #next_lineno = self.get_next_var_num(node.targets[0].id)

                    self.add_to_graph(node, node.targets[0].id, (MC2_Types.CLEAR, ""))

                    if fn_name in ("c_fix_mat", "cfixMatrix"):
                        #rows = node.value.args[0].n
                        #cols = node.value.args[1].n
                        #self.mat_to_dim[(node.targets[0].id, next_lineno)]
                        self.process_mat_declare(node.targets[0].id, self.var_to_count[node.targets[0].id], node.value.args)


                # HACKY Way just in case another case didn't consider, basically this is calling library functions.
                else:
                    if isinstance(node.value.func, ast.Attribute):
                        self.visit_Call(node.value)
                    else:
                        print "LIBRARY FUNCTION: ", node.value.func.id
                        if len(node.value.args) > 1:
                            res_type = self.check_types(node.value.args)
                        else:
                            # Hack
                            if len(node.value.args) == 0:
                                print "0 PARAMETER FUNCTION CALL. Skip for now!"
                                res_type = (MC2_Types.SECRET, "")
                                return
                            elif isinstance(node.value.args[0], ast.Num):
                                res_type = (MC2_Types.CLEAR, "")
                            else:
                                res_type = self.lookup_type(node.value.args[0].id)

                        # Plan on supporting sum and basic reduce functions. At least add and mul.
                        if fn_name in self.aggregation_fns:
                            iter_name = node.value.args[0].id
                            latest_ref = self.lookup(iter_name)
                            lst_types = self.lst_to_types[latest_ref]
                            lst_res_type = self.check_types([ast.Name(id=ele[0][0]) for ele in lst_types])
                            # Maps each party to the list of inputs that are being aggregated.
                            private_party_to_inputs = {}
                            for i in range(len(lst_types)):
                                ref, mc2_type = lst_types[i]
                                private_party = mc2_type[1]
                                if private_party not in private_party_to_inputs.keys():
                                    private_party_to_inputs[private_party] = []
                                private_party_to_inputs[private_party].append(ref)

                            lst_sublists = []
                            for private_party in private_party_to_inputs.keys():
                                lst_private_inputs = private_party_to_inputs[private_party]
                                name_sublist = "{0}_{1}".format(fn_name, private_party)
                                sublist_lineno = self.get_next_var_num(name_sublist)
                                elts = [ast.Name(id=ele[0] + str(ele[1])) for ele in lst_private_inputs]
                                #Add in a node that is like sub_list = [priv1_a, priv1_b, ... ]
                                ast_node = ast.Assign(targets=[ast.Name(id=name_sublist)], value=ast.List(elts=elts))
                                #self.var_graph.add_node((name_sublist, sublist_lineno), op=ast_node, mc2_type=(MC2_Types.PRIVATE, private_party))

                                #for ref_to_input in lst_private_inputs:
                                    #self.var_graph.add_edge(ref_to_input, (name_sublist, sublist_lineno))


                                #lst_sublists.append((name_sublist, sublist_lineno))

                            self.add_to_graph(node, node.targets[0].id, lst_res_type)
                            if fn_name not in self.reduce_fn:
                                self.aggregations[(fn_name, node.targets[0].id, self.var_to_count[node.targets[0].id])] = private_party_to_inputs
                            else:
                                lambda_func = node.value.args[1]
                                self.aggregations[((fn_name, lambda_func), node.targets[0].id, self.var_to_count[node.targets[0].id])] = private_party_to_inputs


                            print "Aggregation: ", self.var_to_count[node.targets[0].id]

                            # Hack!!! Treat lists as 1-d column vectors.

                            # Add edges between each private var to their own local list.
                            """
                            copy_aggregation_node = copy.deepcopy(node)
                            copy_aggregation_node.args = [ast.Name(id=sublist_ref[0]) for sublist_ref in lst_sublists]
                            #self.var_graph.add_node((iter_name, iter_new_lineno), op=copy_aggregation_node, mc2_type=lst_res_type)
                            for sublist_ref in lst_sublists:
                                self.var_graph.add_edge(sublist_ref, (iter_name, self.var_to_count[iter_name]))




                            self.var_graph.add_edge((iter_name, self.var_to_count[iter_name]), (node.targets[0].id, self.var_to_count[node.targets[0].id]))

                            """

                        else:
                            # All the matrix-related library functions.
                            rows, cols = self.track_dim_library_fn(fn_name, node.targets[0].id, node.value.args)
                            next_lineno = self.get_next_var_num(node.targets[0].id)
                            self.mat_to_dim[(node.targets[0].id, next_lineno)] = (rows, cols)
                            # print "Library fn: {0} outputs this: {1}".format(fn_name, node.targets[0].id), rows, cols

                            self.var_tracker[(node.targets[0].id, next_lineno)] = res_type
                            # print node.targets[0].id, res_type
                            print "Var: {0} has type: {1}".format(node.targets[0].id, res_type)
                            # Track dependencies between input and output
                            self.track_input_output_dependencies(node.value.args, [node.targets[0]], node, res_type)



            # a = b, where b is another variable
            elif isinstance(node.value, ast.Name) and not isinstance(node.targets[0], ast.Subscript):
                # Sometimes you have arr[i][j] = ... and I guess we're not doing copying or containers, that'd be very difficult.
                #print "Assign name: {0} to value: {1}".format(node.targets[0].id, node.value.id)
                #print "Check value is correct type. Val: {0}, type: {1}".format(node.value.id, self.lookup_type(node.value.id))
                if node.value.id in self.python_clear:
                    mc2_type = (MC2_Types.CLEAR, "")
                    next_lineno = self.get_next_var_num(node.targets[0].id)

                else:
                    mc2_type = self.lookup_type(node.value.id)
                    # Check dimensions. Have to avoid cases with python types like "True" and "False"
                    latest_ref = self.lookup(node.value.id)
                    next_lineno = self.get_next_var_num(node.targets[0].id)
                    if latest_ref in self.mat_to_dim.keys():
                        self.mat_to_dim[(node.targets[0].id, next_lineno)] = self.mat_to_dim[latest_ref]




                self.var_tracker[(node.targets[0].id, next_lineno)] = mc2_type

                if self.name_exists(node.value.id):
                    rhs_var_name = self.lookup(node.value.id)
                    self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=mc2_type, op=node)
                    self.var_graph.add_edge(rhs_var_name, (node.targets[0].id, next_lineno))
                    self.name_to_node[(node.targets[0].id, next_lineno)] = node






            # a = b + c
            elif isinstance(node.value, ast.BinOp):
                names = set()
                #left_name = self.get_subscript_name(node.value.left, names)
                #right_name = self.get_subscript_name(node.value.right)
                self.get_binop_name(node.value, names)
                print "NAMES: ", names
                #res_type = self.check_types([ast.Name(id=left_name), ast.Name(id=right_name)])
                res_type = self.check_types([ast.Name(id=name) for name in list(names)])
                #print "Subscript left name: {0}, left_type: {1}, right name: {2}, right type: {3}, res_type:{4}".format(left_name, self.lookup_type(left_name), right_name, self.lookup_type(right_name), res_type)
                next_lineno = self.get_next_var_num(node.targets[0].id)
                self.var_tracker[(node.targets[0].id, next_lineno)] = res_type


                for name in names:
                    try:
                        if self.name_exists(name):
                            ref_to_rhs = self.lookup(name)
                            self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=res_type, op=node)
                            self.var_graph.add_edge(ref_to_rhs, (node.targets[0].id, next_lineno))


                    except ValueError as e:
                        pass
                        #print e
                        #print "Probably encountered an ast.Num object"

                self.name_to_node[(node.targets[0].id, next_lineno)] = node
                """
                try:

                    if self.name_exists(left_name):
                        ref_to_rhs = self.lookup(left_name)
                        self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=res_type, op=node)
                        self.var_graph.add_edge(ref_to_rhs, (node.targets[0].id, next_lineno))
                    if self.name_exists(right_name):
                        ref_to_rhs = self.lookup(right_name)
                        self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=res_type, op=node)
                        self.var_graph.add_edge(ref_to_rhs, (node.targets[0].id, next_lineno))

                    self.name_to_node[(node.targets[0].id, next_lineno)] = node
                except ValueError as e:
                    print e
                    print "Probably encountered an ast.Num object"
                """


            elif isinstance(node.value, ast.List):
                mc2_type = (MC2_Types.SECRET, "")
                #self.var_tracker[(node.targets[0].id, next_lineno)] = mc2_type
                #self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=mc2_type, op=node)
                #next_lineno = self.get_next_var_num(node.targets[0].id)
                self.add_to_graph(node, node.targets[0].id, mc2_type)

                lst_name = node.targets[0].id
                self.mat_to_dim[(lst_name, self.var_to_count[lst_name])] = []


                key_to_lst = (lst_name, self.var_to_count[lst_name])
                self.lst_to_types[key_to_lst] = []

                for ele in node.value.elts:
                    if isinstance(ele, ast.Name):
                        ele_name = ele.id
                        self.var_graph.add_edge((ele_name, self.var_to_count[ele_name]), (node.targets[0].id, next_lineno))

                        ref_to_element = self.lookup(ele_name)
                        ele_type = self.lookup_type(ele_name)
                        self.lst_to_types[key_to_lst].append(((ele_name, self.var_to_count[ele_name]), ele_type))
                        lineno = self.var_to_count[node.targets[0].id]
                        if ref_to_element in self.mat_to_dim.keys():
                            self.mat_to_dim[(node.targets[0].id, lineno)].append(self.mat_to_dim[ref_to_element])
                        else:
                            self.mat_to_dim[(node.targets[0].id, lineno)].append(None)

                    elif isinstance(ele, ast.Num):
                        raise ValueError("Don't put constants in lists. Just screws everything up.")




            elif isinstance(node.value, ast.Subscript):
                subscript_obj = node.value
                subscript_name = self.get_subscript_name(subscript_obj)
                mc2_type = self.lookup_type(subscript_name)
                next_lineno = self.get_next_var_num(node.targets[0].id)
                self.var_tracker[(node.targets[0].id, next_lineno)] = mc2_type
                if self.name_exists(subscript_name):
                    self.var_graph.add_node((node.targets[0].id, next_lineno), mc2_type=mc2_type, op=node)
                    self.var_graph.add_edge((subscript_name, self.var_to_count[subscript_name]), (node.targets[0].id, next_lineno))
                    self.name_to_node[(node.targets[0].id, next_lineno)] = node

                ref_to_subscript = self.lookup(subscript_name)
                if ref_to_subscript in self.mat_to_dim.keys():
                    if isinstance(subscript_obj.slice.value, ast.Num) and isinstance(self.mat_to_dim[ref_to_subscript], list):
                        index = subscript_obj.slice.value.n
                        mat_dim = self.mat_to_dim[ref_to_subscript][index]
                        if mat_dim != None:
                            self.mat_to_dim[(node.targets[0].id, next_lineno)] = mat_dim

    # Track dimensions of amtrix from library calls
    def track_dim_library_fn(self, fn_name, name, args):
        if fn_name in ("matinv", "matadd", "matsub", "placeholder"):
            first_arg = args[0].id
            latest_ref = self.lookup(first_arg)
            #self.mat_to_dim[(name, lineno)] = self.mat_to_dim[latest_ref]
            return self.mat_to_dim[latest_ref]

        elif fn_name in ("transpose", "placeholder"):
            first_arg = args[0].id
            latest_ref = self.lookup(first_arg)
            rows, cols = self.mat_to_dim[latest_ref]
            return (cols, rows)
        elif fn_name in ("mat_const_mul", "placeholder"):
            second_arg = args[1].id
            latest_ref = self.lookup(second_arg)
            #self.mat_to_dim[(name, lineno)] = self.mat_to_dim[latest_ref]
            return self.mat_to_dim[latest_ref]
        elif fn_name in ("matmul", "placeholder"):
            # DO SHIT HERE???????
            first_arg = args[0].id
            second_arg = args[1].id
            latest_ref_first = self.lookup(first_arg)
            latest_ref_second = self.lookup(second_arg)
            left_rows, left_cols = self.mat_to_dim[latest_ref_first]
            right_rows, right_cols = self.mat_to_dim[latest_ref_second]
            #self.mat_to_dim[(name, lineno)] = (left_rows, right_cols)
            key = (left_rows, left_cols, right_rows, right_cols, 'sfix')
            if key not in self.vectorized_calls.keys():
                self.vectorized_calls[key] = 0


            self.vectorized_calls[key] += 1

            return (left_rows, right_cols)
        else:
            print "Matrix library function name", fn_name, name
            raise ValueError

    def get_subscript_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        subscript_obj = node
        while isinstance(subscript_obj, ast.Subscript):
            subscript_obj = subscript_obj.value

        return subscript_obj.id


    # Get the set of names in a binary operation.
    def get_binop_name(self, node, s):
        if isinstance(node, ast.Name):
            s.add(node.id)
            return
        elif isinstance(node, ast.Num):
            return
        elif isinstance(node, ast.Subscript):
            s.add(self.get_subscript_name(node))
            return
        left_name = self.get_binop_name(node.left, s)
        right_name = self.get_binop_name(node.right, s)
        if left_name:
            s.add(left_name)
        if right_name:
            s.add(right_name)
        return






    # Add edges between private inputs (if they exist) and the outputs of a function call.
    def track_input_output_dependencies(self, lst_args, lst_retvals, node, res_type):
        for retval in lst_retvals:
            for arg in lst_args:
                try:
                    if self.name_exists(arg.id):
                        ref_to_arg = self.lookup(arg.id)
                        next_lineno = self.var_to_count[retval.id]
                        self.var_graph.add_node((retval.id, next_lineno), mc2_type=res_type, op=node)
                        self.var_graph.add_edge(ref_to_arg, (retval.id, next_lineno))
                        self.name_to_node[(retval.id, next_lineno)] = node

                except AttributeError as e:
                    #print e
                    #print "Exception: Mapping input types to output types. ast Num Object"
                    pass


    # Check if a call node is a private input. Basically checks if the expression is of the form "*.read_input" or something of the sort.
    def check_private_input(self, node):
        return isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in self.private_inputs



    def get_fn_name(self, node):
        if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
            return node.func.id

    # Returns the "most recent" reference to name. Returns the name along with line number of this variable.
    def lookup(self, name):
        lst_matching_names = []
        for k in self.var_tracker.keys():
            if k[0] == name:
                lst_matching_names.append(k)

        if not lst_matching_names:
            raise ValueError("No name: {0}".format(name))

        # Return the latest instance of name.
        return sorted(lst_matching_names, key=lambda item: -item[1])[0]


    def name_exists(self, name):
        lst_matching_names = []
        for k in self.var_tracker.keys():
            if k[0] == name:
                return True

        return False

    # Look up the mc2_type for name
    def lookup_type(self, name):
        target_node = self.lookup(name)
        lst_nodes = list(self.var_graph.nodes(data=True))
        for node in lst_nodes:
            if target_node == node[0]:
                return node[1]['mc2_type']

        return (MC2_Types.SECRET, "")


    # Gets the next counter for name. Stores it in var_to_count dictionary
    def get_next_var_num(self, name):
        self.var_to_count[name] = self.global_counter
        self.global_counter += 1
        return self.var_to_count[name]


    # Given a list of MC2 input types (usually arguments) output the correct MC2 output type according to the paper.
    def check_types(self, lst_args):
        party = None
        for arg in lst_args:
            # HARDCODE, matmul so far requires 'sfix' as an argument which this function doesn't recognize, ugh.
            if not isinstance(arg, ast.Num):
                if arg.id not in self.secret_types:
                    #print "Arg: {0}, type: {1}".format(arg.id, self.lookup_type(arg.id))
                    lookup_type = self.lookup_type(arg.id)
                    if lookup_type[0]  == MC2_Types.SECRET:
                        return (MC2_Types.SECRET, "")
                    elif lookup_type[0] == MC2_Types.PRIVATE:
                        if party is None:
                            party = lookup_type[1]
                        elif lookup_type[1] != party:
                            return (MC2_Types.SECRET, "")

        if party is not None:
            return (MC2_Types.PRIVATE, party)
        else:
            return (MC2_Types.CLEAR, "")



    def postprocess(self):
        for name, lineno in sorted(self.name_to_node.keys(), key=lambda item: item[1]):
            self.name_to_node[(name, lineno)].lineno = lineno


        #print "Postprocessing graph, adding line-numbers"
        #for name, lineno in sorted(self.name_to_node.keys(), key=lambda item: item[1]):
            #print name, lineno, self.name_to_node[(name, lineno)].lineno




    def trace_private_computation(self, party):
        # List of local
        lst_local_ops = []
        lst_local_nodes = set()
        # List of operations that you want to "get rid of" in the main program.
        #lst_private = []
        lst_private = []
        secret_to_dependents = {}
        d = {}
        print "NUMBER OF NODES: ", len(self.var_graph.nodes())
        for node in self.var_graph.nodes(data=True):
            print node
            d[node[0]] = node


        for node in self.var_graph.nodes():
            dict_entry = d[node]
            mc2_type =  dict_entry[1]['mc2_type']
            if mc2_type[0] in (MC2_Types.CLEAR, MC2_Types.PRIVATE):
                heapq.heappush(self.pq, (node[1], node))

        while self.pq:
            priority, node = heapq.heappop(self.pq)
            node_type = d[node][1]['mc2_type']
            print "Node type: ", node_type

            if node_type[0] == MC2_Types.SECRET and self.check_if_source_private(node):
                lst_local_ops.append(d[node])
                # lst_private.append(d[node])
                if self.var_graph.in_edges(node):
                    lst_priv_dependents = []
                    for source, target in self.var_graph.in_edges(node):
                        source_type = d[source][1]['mc2_type']
                        if source_type[0] in (MC2_Types.PRIVATE, "Placeholder"):
                            source_party = source_type[1]
                            priv_dependent = ((source[0], source[1]), source_party)
                            lst_priv_dependents.append(priv_dependent)

                    secret_to_dependents[node] = lst_priv_dependents
            elif node_type[0] != MC2_Types.SECRET:
                check_private = (node_type[0] == MC2_Types.PRIVATE and node_type[1] == party)
                #print "CHECK PRIVATE: ", check_private, node_type, node_type[1], party
                if node_type[0] == MC2_Types.PRIVATE:

                    lst_private.append(d[node])


                if check_private or node_type[0] == MC2_Types.CLEAR:
                    if d[node][1]['op'] not in lst_local_nodes: #and self.var_graph.in_edges(d[node][0]):
                        lst_local_ops.append(d[node])
                        lst_local_nodes.add(d[node][1]['op'])


                for source, target in self.var_graph.out_edges(node):
                    heapq.heappush(self.pq, (target[1], target))





        return lst_local_ops, secret_to_dependents, lst_private, d


    def insert_local_compute(self, lst_clear_private, secret_to_dependents, party, node_dict):

        # Sort secret nodes by k
        #print "List local: ", lst_clear_private
        sorted_secret_keys = sorted(secret_to_dependents.keys(), key=lambda item: item[1])

        lst_to_insert = []
        lst_secret_nodes = []
        for i in range(len(sorted_secret_keys)):
            secret_node = sorted_secret_keys[i]
            secret_name = secret_node[0]
            secret_lineno = secret_node[1]

            lst_secret_nodes.append(secret_node)
            lst_secret_dependents = secret_to_dependents[secret_node]
            # Assume each secret node has only 1 private dependent.
            private_dependent, source_party = lst_secret_dependents[0]

            if source_party == party:
                # Extracts the variable name of the secret node's dependents.
                lst_secret_dependents = [e[0][0] for e in lst_secret_dependents]
                # Get the line number of the latest reference to any of the dependents to this secret node
                # lst = [item[0][1] if (item[0][1] < secret_lineno and item[0][0] in lst_secret_dependents and item[1]['mc2_type'][1] == party) else -float('inf') for item in lst_clear_private]
                clear_node = max(lst_clear_private, key=lambda item: item[0][1] if (item[0][1] < secret_lineno and item[0][0] in lst_secret_dependents and item[1]['mc2_type'][1] == party) else -float('inf'))
                clear_index = lst_clear_private.index(clear_node)
                lst_to_insert.append((clear_node, secret_node))

                print "CLEAR NODE: ", clear_node



        for node, secret_node in lst_to_insert:
            index = lst_clear_private.index(node)
            secret_node_dependent = secret_to_dependents[secret_node]
            secret_name = secret_node_dependent[0][0][0]
            ast_node = ast.Call(starargs=None, kwargs=None, keywords=[], func=ast.Attribute(attr="append", value=ast.Name(id="data")), args=[ast.Name(id=secret_name)])

            #ast.Assign(targets=[ast.Name(id='data')], value=ast.BinOp(left=ast.Name(id='data'), right=ast.Name(id=secret_node_dependent[0][0][0]), op=ast.Add()))
            d = {'op':ast_node, 'mc2_type': (MC2_Types.PRIVATE, party)}
            lst_clear_private.insert(index + 1, (None, d))


        # Insert aggregation
        # fn_name is the name of the aggregation function, var_name is the variable name assigned to the result.
        for fn_name, var_name, aggr_lineno in self.aggregations.keys():
            lst_private = self.aggregations[(fn_name, var_name, aggr_lineno)][party]
            if isinstance(fn_name, tuple):
                lst_name = "{0}_{1}".format(fn_name[0], aggr_lineno)
            else:
                lst_name = "{0}_{1}".format(fn_name, aggr_lineno)
            max_index = -1
            for private_aggr in lst_private:
                node = node_dict[private_aggr]
                index = lst_clear_private.index(node)
                arg_name = node[0][0]

                ast_node = ast.Call(starargs=None, kwargs=None, keywords=[], func=ast.Attribute(attr="append", value=ast.Name(id=lst_name)), args=[ast.Name(id=arg_name)])
                d = {'op': ast_node, 'mc2_type': (MC2_Types.PRIVATE, party)}
                lst_clear_private.insert(index + 1, (None, d))
                max_index = max(max_index, lst_clear_private.index((None, d)) + 1)

            # Insert final aggregation: Ex: res = sum(lst)
            # The idea is want to insert this at the very end.
            if isinstance(fn_name, tuple):
                aggregation_node = ast.Assign(targets=[ast.Name(id=var_name)], value=ast.Call(args=[ast.Name(id=lst_name), fn_name[1]], func=ast.Name(id=fn_name[0]), keywords=[], kwargs=None, starargs=None))
            else:
                aggregation_node = ast.Assign(targets=[ast.Name(id=var_name)], value=ast.Call(args=[ast.Name(id=lst_name)], func=ast.Name(id=fn_name), keywords=[], kwargs=None, starargs=None))
            d = {'op': aggregation_node, 'mc2_type': (MC2_Types.PRIVATE, party)}
            lst_clear_private.insert(max_index + 1, (None, d))

            # Add the aggregation result to "data". data.append(sum_res)
            ast_node = ast.Call(starargs=None, kwargs=None, keywords=[], func=ast.Attribute(attr="append", value=ast.Name(id="data")), args=[ast.Name(id=var_name)])
            d = {'op':ast_node, 'mc2_type': (MC2_Types.PRIVATE, party)}
            lst_clear_private.insert(max_index + 2, (None, d))


        local_program = self.write_local_program(lst_clear_private, party)


        #lst_secret_nodes.extend([e[1] for e in lst_to_insert])
        print "LIST SECRET: ", lst_secret_nodes
        return lst_clear_private, lst_secret_nodes, local_program



    def write_local_program(self, lst_clear_private, party):
        #temp_mpc_type = mpc_type
        #mpc_type = LOCAL
        s = ""
        s += "pmat = p_mat()\n"
        s += "pmat.preprocess()\n"
        s += "data = []\n"
        for fn_name, var_name, lineno in self.aggregations.keys():
            if isinstance(fn_name, tuple):
                s+= "{0}_{1} = []\n".format(fn_name[0], lineno)
            else:
                s+= "{0}_{1} = []\n".format(fn_name, lineno)
        for node in lst_clear_private:
            print "Private node: ", astunparse.unparse(node[1]['op'])
            mc2_type = node[1]['mc2_type']
            check_private = (mc2_type[0] == MC2_Types.PRIVATE and mc2_type[1] == party)
            if check_private or mc2_type[0] == MC2_Types.CLEAR:
                ast_node = node[1]['op']
                if mc2_type[0] == MC2_Types.PRIVATE:
                    copy_ast_node = copy.deepcopy(ast_node)
                    if isinstance(copy_ast_node, ast.Assign):
                        res = self.rename_private(copy_ast_node.value)
                    else:
                        res = self.rename_private(copy_ast_node)
                    if res != None:
                        copy_ast_node.value = res

                    ast_node = copy_ast_node

                if ast_node:
                    s += astunparse.unparse(ast_node)



        s += "write_private_data(data)\n"

        print "LOCAL PROGRAM"
        print s
        return s

    # For precomputation, rename s_fix_mat.read_input ----> pmat.read_input
    def rename_private(self, call_node):
        if not isinstance(call_node, ast.Call):
            return

        if not isinstance(call_node.func, ast.Attribute):
            print "NO ATTRIBUTE?", call_node.func.__dict__
        else:
            print "CALL NODE", call_node.func.value.id
            copy_node = copy.deepcopy(call_node)
            copy_node.func.value.id = "pmat"
            return copy_node





    # Only add secret nodes that has a direct dependence on
    def check_if_source_private(self, secret_target):
        for node in self.var_graph.nodes(data=True):
            mc2_type = node[1]['mc2_type']
            if mc2_type[0] == MC2_Types.PRIVATE:
                if nx.has_path(self.var_graph, node[0], secret_target):
                    return True
        return False





class ProgramSplitter(ast.NodeTransformer):
    def __init__(self, lst_clear_private, lst_secret, secret_to_dependents, mat_to_dim, name_to_node, lst_private, aggregations, party):
        self.lst_clear_private = lst_clear_private
        self.lst_secret = lst_secret
        self.secret_to_dependents = secret_to_dependents
        self.mat_to_dim = mat_to_dim
        self.name_to_node = name_to_node
        self.lst_private = lst_private
        self.party = party
        self.aggregations = aggregations
        # Map the reduce function to the number of arguments it might need.
        self.reduce_map = {"sum": 1, "sum_lib": 1, "min_lib": 1, "max_lib": 1, "avg": 2, "reduce_lib": 1}

    def visit_FunctionDef(self, node):
        return


    def visit_Assign(self, node):
        if hasattr(node, 'lineno'):
            aggr = self.is_aggregation(node)
            if aggr:
                fn_name, var_name, lineno = aggr
                lst_assign = []
                lst_names = []
                private_party_to_inputs = self.aggregations[aggr]
                for private_party in private_party_to_inputs.keys():
                    lst_private_inputs = private_party_to_inputs[private_party]
                    if isinstance(fn_name, tuple):
                        call_node = ast.Call(kwargs=None, starargs=None, keywords=[], func=ast.Attribute(attr="read_input", value=ast.Name(id="s_fix_mat")), args=[ast.Num(n=self.reduce_map[fn_name[0]]), ast.Num(n=1), ast.Num(n=private_party)])
                        aggr_private_name = "{0}_{1}_{2}".format(fn_name[0], lineno, private_party)
                    else:
                        call_node = ast.Call(kwargs=None, starargs=None, keywords=[], func=ast.Attribute(attr="read_input", value=ast.Name(id="s_fix_mat")), args=[ast.Num(n=self.reduce_map[fn_name]), ast.Num(n=1), ast.Num(n=private_party)])
                        aggr_private_name = "{0}_{1}_{2}".format(fn_name, lineno, private_party)

                    temp_assign = ast.Assign(targets=[ast.Name(id=aggr_private_name)], value=call_node)
                    lst_assign.append(temp_assign)
                    lst_names.append(aggr_private_name)


                if isinstance(fn_name, tuple):
                    aggregation_call_node = ast.Call(func=ast.Name(id=fn_name[0]), args=[ast.List(elts=[ast.Name(id=priv_name) for priv_name in lst_names]), fn_name[1]], keywords=[], starargs=None, kwargs=None)
                else:
                    aggregation_call_node = ast.Call(func=ast.Name(id=fn_name), args=[ast.List(elts=[ast.Name(id=priv_name) for priv_name in lst_names])], keywords=[], starargs=None, kwargs=None)
                aggregation_node = ast.Assign(targets=node.targets, value=aggregation_call_node)
                return lst_assign + [aggregation_node]

            secret_node = self.is_secret(node)
            if secret_node != None:
                # Insert assign
                private_dependent, private_party = self.secret_to_dependents[secret_node][0]
                rows, cols = self.mat_to_dim[private_dependent]
                private_name = "private_input" + str(node.lineno)
                call_node = ast.Call(kwargs=None, starargs=None, keywords=[], func=ast.Attribute(attr="read_input", value=ast.Name(id="s_fix_mat")), args=[ast.Num(n=rows), ast.Num(n=cols), ast.Num(n=private_party)])
                temp_assign = ast.Assign(targets=[ast.Name(id=private_name)], value=call_node)
                return [temp_assign, node]

            if self.is_private(node):
                return []

        return node


    def visit_Call(self, node):

        if hasattr(node, 'lineno'):

            secret_node = self.is_secret(node)

            if secret_node != None:
                # Insert assign
                private_dependent, private_party = self.secret_to_dependents[secret_node][0]
                if private_dependent in self.mat_to_dim.keys():
                    rows, cols = self.mat_to_dim[private_dependent]
                else:
                    # TODO: HACK, I REPEAT THIS IS A HACK
                    # Assume values are just 1 by 1 matrices.
                    rows, cols = (1, 1)
                private_name = "private_input" + str(node.lineno)

                call_node = ast.Call(kwargs=None, starargs=None, keywords=[], func=ast.Attribute(attr="read_input", value=ast.Name(id="s_fix_mat")), args=[ast.Num(n=rows), ast.Num(n=cols), ast.Num(n=private_party)])
                temp_assign = ast.Assign(targets=[ast.Name(id=private_name)], value=call_node)
                node.args[0].id = private_name
                new_assign = ast.Assign(targets=[ast.Name(id="throwaway")], value=node)


                source = astor.to_source(temp_assign)
                print "TEMP ASSIGN: ", source
                source = astor.to_source(new_assign)
                print "NEW ASSIGN: ", source
                return [temp_assign, new_assign]

            if self.is_private(node):
                return []

        return node


    def is_secret(self, node):
        for secret_node in self.lst_secret:
            if secret_node[1] == node.lineno:
                return secret_node

        return None



    def is_private(self, node):
        lineno = node.lineno
        for priv_node in self.lst_private:
            if priv_node[0][1] == lineno:
                return True

        return False

    def is_aggregation(self, node):
        for fn_name, var_name, lineno in self.aggregations.keys():
            if node.lineno == lineno:
                return (fn_name, var_name, lineno)

        return False


class CountMatmulHelper(ast.NodeVisitor):

    def __init__(self, mat_to_dim):
        self.mat_to_dim = mat_to_dim

        self.vectorized_calls = {}


    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):

            if node.value.func.id == "matmul":
                left_rows = node.value.args[2].n
                left_cols = node.value.args[3].n
                right_rows = node.value.args[4].n
                right_cols = node.value.args[5].n
                mat_type = node.value.args[6].id
                if (left_rows, left_cols, right_rows, right_cols, mat_type) not in self.vectorized_calls.keys():
                    self.vectorized_calls[(left_rows, left_cols, right_rows, right_cols, mat_type)] = 0
                self.vectorized_calls[(left_rows, left_cols, right_rows, right_cols, mat_type)] += 1



import loop_unroll
import inline

class ASTParser(object):

    def __init__(self, fname, party, debug=False):
        f = open(fname, 'r')
        s = f.read()
        f.close()
        if mpc_type == SPDZ:
            header = "open_channel(0)\n"
            #header += "pmat = p_mat()\n"
            #header += "pmat.preprocess()\n"
            s = header + s + "\nclose_channel(0)\n"

        self.tree = ast.parse(s)
        self.filename = fname
        self.source = s
        self.debug = debug
        self.party = int(party)

    def parse(self):
        # Run through a bunch of parsers
        # hardcoded to count the # of matmul calls. Returns the # of matmul calls.
        target = "matmul"
        # Try inlining
        #self.tree = ASTChecks().visit(self.tree)
        #
        #source = astor.to_source(self.tree)
        #print "AFTER IF CHECKS: "
        #print source


        dep = ProcessDependencies()
        dep.visit(self.tree)

        #helper = inline.RenameVisitorHelper(dep.fns_to_params)
        #helper.visit(self.tree)
        #rename = inline.RenameVisitor(dep.fns_to_params, helper.fns_to_vars)
        #self.tree = rename.visit(self.tree)
        #inliner = inline.InlineSubstitution(rename.fn_to_node, rename.fns_to_params, dep.G)
        #self.tree = inliner.visit(self.tree)
        #self.tree = ConstantPropagation().visit(self.tree)
        #self.tree = loop_unroll.UnrollStep().visit(self.tree)
        # After for-loops unroll, propagate the for-loop constants

        #self.tree = ConstantPropagation().visit(self.tree)
        # Another pass to fix all the subscripts. Ex: a[i] => a[0] for example
        #self.tree = ConstantPropagation().visit(self.tree)

        #s = PrivateTypeInference(rename.fn_to_node, rename.fns_to_params, dep.G)
        #s.visit(self.tree)
        #source = astor.to_source(self.tree)

        #print "After unrolling: ", source

        #s = ProgramSplitterHelper()
        #s.visit(self.tree)
        # Map matrices to their dimensions as well. SHIET
        # print "MATRIX DIMENSIONS: ", s.mat_to_dim
        # print "Count matmul in program splitter: ", s.vectorized_calls, haven't separated out local computation yet.

        #lst_local_ops, secret_to_dependents, lst_private, d = s.trace_private_computation(self.party)
        #lst_clear_private, lst_secret_nodes, local_program = s.insert_local_compute(lst_local_ops, secret_to_dependents, self.party, d)

        #s.postprocess()

        #splitter = ProgramSplitter(lst_local_ops, lst_secret_nodes, secret_to_dependents, s.mat_to_dim, s.name_to_node, lst_private, s.aggregations, self.party)
        #self.tree = splitter.visit(self.tree)

        #count_matmul = CountMatmulHelper(s.mat_to_dim)
        #count_matmul.visit(self.tree)
        #print "Count matmul: ", count_matmul.vectorized_calls


        #self.tree = ForLoopParser().visit(self.tree)
        #self.tree = ASTChecks().visit(self.tree)

        #return count_matmul.vectorized_calls, local_program
        return {}, ""

    def execute(self, context):
        source = astor.to_source(self.tree)
        if self.debug:
            print(source)


        #print context
        exec(source, context)




        #exec(compile(self.tree, filename="<ast>", mode="exec"), context)


    def execute_local(self, local_str, context):
        print "LOCAL COMPUTE"
        print local_str
        #temp_mpc_type = interface.mpc_type
        #interface.mpc_type = LOCAL
        local = {}
        #exec(local_str, context, local)
        #print "Global: ", context
        #print "Local: ", local
        #interface.mpc_type = temp_mpc_type
