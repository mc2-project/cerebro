from types import *
from types_gc import *
import compilerLib, library
import symtable
import re
import checker

SPDZ = 0
GC = 1

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
        

class ClearFixedPointFactory(object):
    def __call__(self, value):
        if mpc_type == SPDZ:
            return cfix(value)
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

class SecretFixedPointMatrixFactory(object):
    def __call__(self, rows, columns):
        if not isinstance(rows, int) or not isinstance(columns, int):
            raise ValueError("Matrix sizes must be publicly known integers")
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

ClearInteger = ClearIntegerFactory()
SecretInteger = SecretIntegerFactory()
ClearFixedPoint = ClearFixedPointFactory()
SecretFixedPoint = SecretFixedPointFactory()
SecretFixedPointArray = SecretFixedPointArrayFactory()
SecretFixedPointMatrix = SecretFixedPointMatrixFactory()

compilerLib.VARS["c_int"] = ClearInteger
compilerLib.VARS["s_int"] = SecretInteger
compilerLib.VARS["c_fix"] = ClearFixedPoint
compilerLib.VARS["s_fix"] = SecretFixedPoint
compilerLib.VARS["s_fix_array"] = SecretFixedPointArray
compilerLib.VARS["s_fix_mat"] = SecretFixedPointMatrix
compilerLib.VARS["reveal_all"] = reveal_all


import ast
import astor
from pprint import pprint
import networkx as nx
import operator as op
import astunparse


# Used to convert expressions with binary operators like 5 * 6 into values.
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

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
            print "Private type inference, visit function def: ", node.name
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
        self.G = G
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
                    print "Postprocess: {0}, {1}".format(node, neighbor)
                    self.G[node][neighbor]['weight'] = self.fns_to_calls.get(neighbor, 0) * multiplicative_factor

                if self.G[node][neighbor]['weight'] != 0:
                    print "WEIGHT", node, neighbor, self.G[node][neighbor]['weight']


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

# TODO: Incorporate scope possibly.
class ConstantPropagation(ast.NodeTransformer):
    """NodeTransformer that will inline any Number and String 
    constants defined on the module level wherever they are used
    throughout the module. Any Number or String variable matching [A-Z_]+ 
    on the module level will be used as a constant"""

    def __init__(self):
        self._constants = {}
        #super(ConstantPropagation, self).__init__()

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
        
        #return self._constants.get(node.id, node)
        #print "Visit name: ", node.id
        for k in self._constants.keys():
            if node.id == k:
                print "Name: {0}, value: {1}".format(node.id, self._constants[k].n)
                print node.__dict__
                return self._constants[k]

        return node

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        try:
            # No multiassignment such as a,b = c,d
            if not isinstance(node.value, ast.Tuple):
                val = self.eval_args_helper(node.value)
                node.value = ast.Num(val)
                # So far don't allow multi-assignment, not sure how to go about this.
                self._constants[node.targets[0].id] = node.value
                return node
            else:
                print "Multiassignment not supported: ", node.value.__dict__
        except Exception as e:
            # For some reason, cannot evaluate the right hand side
            print e
            return self.generic_visit(node)

    def eval_args_helper(self, node):
        if hasattr(node, 'n'):
            return node.n
        else:
            left_val = self.eval_args_helper(node.left)
            right_val = self.eval_args_helper(node.right)
            res = operators[type(node.op)](left_val, right_val)
            return res


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
    

class PrivateTypeInference(ast.NodeVisitor):


    def __init__(self, fns_to_nodes, fns_to_params, G):
        self.secret_types = ["sint", "sfix", "s_fix", "s_fix_mat", "sfixMatrix", "sintMatrix"]
        self.clear_types = ["cint", "cfix", "c_fix", "cfixMatrix", "c_fix_mat"]
        self.private_inputs = ["read_input_from", "read_input"]
        # Map function names to function_def nodes
        self.fns_to_nodes = fns_to_nodes
        self.global_scope = "GLOBAL"
        self.scope_stack = []
        # Dependency/Call Graph
        self.G = G
        # Map function to list of names of its parameters.
        self.fns_to_params = fns_to_params
        # Map name and scope (parent) to "type" of object?
        self.var_tracker = {}
        # Maps functions to types of their outputs
        # self.fns_to_outputs = {}

        # Track dependencies between variables. Variables are nodes. 
        # If B depends on A, then draw edge from A -> B
        self.var_graph = nx.DiGraph()
        self.preprocess()


    def preprocess(self):
        topological_ordering = list(nx.topological_sort(self.G))
        print "Topological ordering", topological_ordering

        for node in topological_ordering:
            # Essentially visit the first relevant function that is called by the program. Since python doesn't have a main function like C that gets automatically called.
            if node in self.fns_to_nodes.keys():
                self.scope_stack.insert(0, node)
                self.visit(self.fns_to_nodes[node])
                self.scope_stack.pop(0)
                return

    # Retrieve the parent at the current time from the scope stack. If no parent exists, then return self.global_scope as the parent.
    def get_parent(self):
        if len(self.scope_stack):
            return self.scope_stack[0]
        else:
            return self.global_scope

    def lookup(self, name):
        for scope in self.scope_stack[::-1]:
            if (name, scope) in self.var_tracker.keys():
                return self.var_tracker[(name, scope)]

        return self.var_tracker.get((name, self.global_scope), (MC2_Types.SECRET, ""))


    def visit_FunctionDef(self, node):
        self.scope_stack.insert(0, node.name)
        self.generic_visit(node)
        self.scope_stack.pop(0)



    def visit_For(self, node):
        self.scope_stack.insert(0, node.lineno)
        for item in node.body:
            self.visit(item)
        self.scope_stack.pop(0)


    def visit_Call(self, node):
        parent = self.get_parent()
        if hasattr(node.func, "id") and node.func.id in self.fns_to_nodes.keys():
            fn_name = self.get_fn_name(node)
            print "Calling function: ", fn_name
            # Bind parameters to arguments
            self.bind_params_to_args(node)
            self.scope_stack.insert(0, fn_name)
            self.visit(self.fns_to_nodes[fn_name])
            self.scope_stack.pop(0)
            # if output of function is private.
            print "Function: {0}, has output: {1}".format(fn_name, self.var_tracker[(fn_name, fn_name)])
        else:
            # Probably a library function, so check types of args.
            pass


    # Bind the types of arguments to the parameters of the function so when we enter the function, we actually know all the variables and what is going on.
    def bind_params_to_args(self, node):
        #fn_name = node.func.id
        fn_name = self.get_fn_name(node)
        parent = self.get_parent()
        lst_params = self.fns_to_params[fn_name]
        for i in range(len(lst_params)):
            arg = node.args[i]
            param = lst_params[i]
            if isinstance(arg, ast.Num):
                self.var_tracker[(param, parent)] = (MC2_Types.CLEAR, "")
            else:
                mc2_type = self.lookup(arg.id)
                self.var_tracker[(param, parent)] = mc2_type
                # Add dependency edge between parameter of function and argument passed in.
                if (arg.id, parent) in self.var_graph:
                    # No real operation here, just binding parameters to arguments. How to account for this in the actual code?
                    self.var_graph.add_node((param, fn_name), mc2_type=mc2_type, op=None)
                    self.var_graph.add_edge((arg.id, parent), (param, fn_name))
                #print "Argument: {0}, parent: {1}".format(arg.id, parent)


    def visit_Return(self, node):
        parent = self.get_parent()
        # Multiple ret vals
        if isinstance(node.value, ast.Tuple):
            lst_types_retvals = []
            for i in range(len(node.value.elts)):
                retval_obj = node.value.elts[i]
                if isinstance(retval_obj, ast.Num):
                    lst_types_retvals.append((MC2_Types.CLEAR, ""))
                else:
                    #print "RETURNING ", retval_obj.id
                    lst_types_retvals.append(self.lookup(retval_obj.id))

            self.var_tracker[(parent, parent)] = lst_types_retvals
        # One ret val
        else:
            retval_obj = node.value
            if isinstance(retval_obj, ast.Num):
                self.var_tracker[(parent, parent)] = (MC2_Types.CLEAR, "")
            else:
                self.var_tracker[(parent, parent)] = self.lookup(retval_obj.id)

        #print "RETURN Function: {0} has retval of type: {1}".format(parent, self.var_tracker[(parent, parent)])


    def get_fn_name(self, node):
        if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
            return node.func.id 



    def visit_Assign(self, node):
        parent = self.get_parent()
        # multi-assignment, figure out later
        if isinstance(node.targets[0], ast.Tuple):
            lst_var_names = []
            for name_obj in node.targets[0].elts:
                lst_var_names.append(name_obj.id)

            if isinstance(node.value, ast.Call):
                fn_name = self.get_fn_name(node.value)
                self.visit(node.value)
                lst_types_retvals = self.var_tracker[(fn_name, fn_name)]
                for i in range(len(lst_var_names)):
                    var_name = lst_var_names[i]
                    retval_type = lst_types_retvals[i]
                    self.var_tracker[(var_name, parent)] = retval_type

                print "jesus: ", fn_name, self.var_tracker[(fn_name, fn_name)]
                lst_ = [x[0] in (MC2_Types.PRIVATE, MC2_Types.CLEAR) for x in self.var_tracker[(fn_name, fn_name)]]
                if any(lst_): 
                    print "jesus", fn_name
                    self.track_input_output_dependencies(node.value.args, [node.targets[0]], self.fns_to_nodes[fn_name], add_to_graph=True)
                else:
                    self.track_input_output_dependencies(node.value.args, [node.targets[0]], self.fns_to_nodes[fn_name], add_to_graph=False)

            # Multiassignment of variables like a,b = c, d, currently not supported.
            elif isinstance(node.value, ast.Tuple):
                right_hand_side_var_names = []
                for name_obj in node.value.elts:
                    right_hand_side_var_names.append(name_obj.id)

                assert(len(lst_var_names) == len(right_hand_side_var_names))
                for i in range(len(lst_var_names)):
                    var_name = lst_var_names[i]
                    rhs_var_name = right_hand_side_var_names[i]
                    self.var_tracker[(var_name, parent)] = self.lookup(rhs_var_name)


        else:
            # Something like a = 5
            if isinstance(node.value, ast.Num):
                self.var_tracker[(node.targets[0].id, parent)] = (MC2_Types.CLEAR, "")
                self.var_graph.add_node((node.targets[0].id, parent), mc2_type=(MC2_Types.CLEAR, ""), op=node)


            # Calling .read_input essentially
            elif self.check_private_input(node):
                if not isinstance(node.value.args[len(node.value.args) - 1], ast.Num):
                    raise ValueError("Last argument of read_input is a party which must be a number!")

                party_num = node.value.args[len(node.value.args) - 1].n 
                self.var_tracker[(node.targets[0].id, parent)] = (MC2_Types.PRIVATE, party_num)
                # Add vertex which represents private node.
                self.var_graph.add_node((node.targets[0].id, parent), mc2_type=(MC2_Types.PRIVATE, party_num), op=node)

            # Calling a function that is NOT a library function
            elif isinstance(node.value, ast.Call):
                # a = sfix(5)
                fn_name = self.get_fn_name(node.value)

                if fn_name in self.secret_types:
                    self.var_tracker[(node.targets[0].id, parent)] = (MC2_Types.SECRET, "")
                # a = cfix(5)
                elif fn_name in self.clear_types:
                    self.var_tracker[(node.targets[0].id, parent)] = (MC2_Types.CLEAR, "")
                    self.var_graph.add_node((node.targets[0].id, parent), mc2_type=(MC2_Types.CLEAR, ""), op=node)
                elif fn_name in self.fns_to_nodes.keys():
                    self.visit(node.value)
                    self.var_tracker[(node.targets[0].id, parent)] = self.lookup(fn_name)
                    print "jesus: ", fn_name, self.var_tracker[(fn_name, fn_name)]
                    if self.var_tracker[(fn_name, fn_name)] in (MC2_Types.PRIVATE, MC2_Types.CLEAR):
                        print "jesus", fn_name
                        self.track_input_output_dependencies(node.value.args, [node.targets[0]], self.fns_to_nodes[fn_name], add_to_graph=True)
                    else:
                        self.track_input_output_dependencies(node.value.args, [node.targets[0]], self.fns_to_nodes[fn_name], add_to_graph=False)

                # HACKY Way just in case another case didn't consider, basically this is calling library functions.
                elif node.value.func.id not in self.fns_to_nodes.keys():
                    print "LIBRARY FUNCTION: ", node.value.func.id
                    if len(node.value.args) > 1:
                        res_type = self.check_types(node.value.args)
                    else:
                        if isinstance(node.value.args[0], ast.Num):
                            res_type = (MC2_Types.CLEAR, "")
                        else:
                            res_type = self.lookup(node.value.args[0].id)

                    self.var_tracker[(node.targets[0].id, parent)] = res_type

                    # Track dependencies between input and output
                    self.track_input_output_dependencies(node.value.args, [node.targets[0]], node, add_to_graph=True)

            # a = b, where b is another variable
            elif isinstance(node.value, ast.Name) and not isinstance(node.targets[0], ast.Subscript):
                # Sometimes you have arr[i][j] = ... and I guess we're not doing copying or containers, that'd be very difficult.
                print "Assign name: {0} to value: {1}".format(node.targets[0].id, node.value.id)
                print "Check value is correct type. Val: {0}, type: {1}".format(node.value.id, self.lookup(node.value.id))
                mc2_type = self.lookup(node.value.id)
                self.var_tracker[(node.targets[0].id, parent)] = mc2_type
                if (node.value.id, parent) in self.var_graph.nodes():
                    self.var_graph.add_node((node.targets[0].id, parent), mc2_type=mc2_type, op=node)
                    self.var_graph.add_edge((node.value.id, parent), (node.targets[0].id, parent))


            # a = b + c
            elif isinstance(node.value, ast.BinOp):
                res_type = self.check_types([node.value.left, node.value.right])
                self.var_tracker[(node.targets[0].id, parent)] = res_type

                try:
                    if (node.value.left.id, parent) in self.var_graph.nodes():
                        self.var_graph.add_node((node.targets[0].id, parent), mc2_type=res_type, op=node)
                        self.var_graph.add_edge((node.value.left.id, parent), (node.targets[0].id, parent))
                    if (node.value.right.id, parent) in self.var_graph.nodes():
                        self.var_graph.add_node((node.targets[0].id, parent), mc2_type=res_type, op=node)
                        self.var_graph.add_edge((node.value.right.id, parent), (node.targets[0].id, parent))
                except ValueError as e:
                    print e
                    print "Probably encountered an ast.Num object"

    # Add edges between private inputs (if they exist) and the outputs.
    def track_input_output_dependencies(self, lst_args, lst_retvals, node, add_to_graph=False):
        parent = self.get_parent()
        for arg in lst_args:
            for retval in lst_retvals:
                try:
                    print "I'm here. Argument: {0}, parent: {1}".format(arg.id, parent)
                    if (arg.id, parent) in self.var_graph.nodes():
                        if add_to_graph:
                            self.var_graph.add_node((retval.id, parent), mc2_type=self.lookup(arg.id), op=node)
                        else:
                            self.var_graph.add_node((retval.id, parent), mc2_type=self.lookup(arg.id), op=None)
                        self.var_graph.add_edge((arg.id, parent), (retval.id, parent))
                except AttributeError as e:
                    #print e
                    print "ast Num Object"

    def check_private_input(self, node):
        # Check if a call node is a private input. Basically checks if the expression is of the form "*.read_input" or something of the sort.
        return isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in self.private_inputs
    

    # Given a list of MC2 input types (usually arguments) output the correct MC2 output type according to the paper.
    def check_types(self, lst_args):
        party = None
        for arg in lst_args:
            # HARDCODE, matmul so far requires 'sfix' as an argument which this function doesn't recognize, ugh.
            if not isinstance(arg, ast.Num):
                if arg.id not in self.secret_types:
                    lookup_type = self.lookup(arg.id)
                    if lookup_type  == (MC2_Types.SECRET, ""):
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


import loop_unroll
import inline
class CodeFlattener(ast.NodeTransformer):
    def __init__(self):
        pass 




class ASTParser(object):
    
    def __init__(self, fname, debug=False):
        f = open(fname, 'r')
        s = f.read()
        f.close()
        if mpc_type == SPDZ:
            s = "open_channel(0)\n" + s + "\nclose_channel(0)\n"
        self.tree = ast.parse(s)
        self.filename = fname 
        self.source = s
        self.debug = debug

    def parse(self):
        # Run through a bunch of parsers
        # hardcoded to count the # of matmul calls. Returns the # of matmul calls.
        target = "matmul"
        # Try inlining
<<<<<<< HEAD
        
=======
        self.tree = ASTChecks().visit(self.tree)
        
        self.tree = ConstantPropagation().visit(self.tree)
>>>>>>> f1956ba932c5886e8ac9d0d9f2ec6899264da60c
        dep = ProcessDependencies()
        dep.visit(self.tree)

        helper = inline.RenameVisitorHelper(dep.fns_to_params)
        helper.visit(self.tree)
        rename = inline.RenameVisitor(dep.fns_to_params, helper.fns_to_vars)
        self.tree = rename.visit(self.tree)
        inliner = inline.InlineSubstitution(rename.fn_to_node, rename.fns_to_params, dep.G)
        self.tree = inliner.visit(self.tree)
        #self.tree = ConstantPropagation().visit(self.tree)
        #self.tree = loop_unroll.UnrollStep().visit(self.tree)
        #self.tree = inliner.visit(self.tree)
        #print 'WHOAH'
        #count_calls = CountFnCall(dep.G, dep.functions, target)
        #count_calls.visit(self.tree)
        #count_calls.postprocess()




        #s = PrivateTypeInference(dep.functions, dep.fns_to_params, dep.G)
        #s.visit(self.tree)


        """
        for k in s.var_tracker.keys():
            if s.var_tracker[k][0] != MC2_Types.PRIVATE:
                s.var_tracker.pop(k)

        print "Var tracker: ", s.var_tracker
        """

        #print "Edges:", s.var_graph.edges()
        """
        print "NODES:"
        lst_data =  s.var_graph.nodes().data()
        d = {i[0]:i[1] for i in lst_data}
        precompute_source = ""
        for node in list(nx.algorithms.dag.topological_sort(s.var_graph)):
        #for node in list(nx.topological_sort(nx.line_graph(s.var_graph))):
            op = d[node]['op']
            if op != None:
                precompute_source += astunparse.unparse(op)
            else:
                print "No OP", node

        print precompute_source
        w = checker.Checker(self.tree, file_tokens=checker.make_tokens(self.source), filename=self.filename)
        w.messages.sort(key=lambda m: m.lineno)
        for warning in w.messages:
            print warning

        """
        #self.tree = loop_unroll.UnrollStep().visit(self.tree)
        #
        #self.tree = s.visit(self.tree)
        #print "GLOBAL NAMES: ", s.global_names
        #self.tree = inline.InlineSubstitution(dep.functions, dep.fns_to_params).visit(self.tree)
        #self.tree = loop_unroll.UnrollStep().visit(self.tree)
        #self.tree = inline.RenameVisitor(dep.fns_to_params).visit(self.tree)
        #print "Number of matmul calls: ", count_calls.counter
        #print "Functions to calls: ", count_calls.fns_to_calls
        #print "Matmul to dimensions: ", count_calls.matmul_to_dims
        #print "Matmul to calls: ", count_calls.matmul_to_calls
        #print "Vectorized Triples requirement: ", count_calls.vectorized_calls

        #self.tree = ForLoopParser().visit(self.tree)
        self.tree = ASTChecks().visit(self.tree)

        #return count_calls.vectorized_calls
        return {}

    def execute(self, context):
        source = astor.to_source(self.tree)
        if self.debug:
            print(source)
        exec(source, context)
        #exec(compile(self.tree, filename="<ast>", mode="exec"), context)
