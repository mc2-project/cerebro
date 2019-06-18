from types import *
from types_gc import *
import compilerLib, library
import ast
import symtable
import re

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


    def visit_FunctionDef(self, node):
        # Not considering differing scopes and same name.
        if node.name != self.target_fn and node.name not in self.mllib_fn:
            self.functions[node.name] = node
        else:
            new_target_fn_name = node.name + str(node.lineno)
            self.G.add_node(new_target_fn_name)
            self.functions[new_target_fn_name] = node 

        print "Visiting function: ", node.name
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
            print "Function {0} called by {1}".format(fn_name, parent)
            self.scope_stack.insert(0, fn_name)
            print "First time visiting: ", fn_name
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
        print "For loop lineno: ", node.lineno
        self.functions[node.lineno] = node
        
        print "Parent of for: ", self.get_parent()

        if node.lineno not in self.G.nodes:
            self.G.add_node(node.lineno)


        parent = self.get_parent()
        print "For loop: Adding edge from: {0} to {1}".format(parent, node.lineno)
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
            print "Function name: ", fn_name
            if self.target_fn in str(fn_name):
                self.fns_to_calls[fn_name] = 1
                self.matmul_to_calls[fn_name] = 0
                self.lst_matmuls.append(fn_name)


        print "List of matmul calls: ", self.lst_matmuls

    def process(self):
        topological_ordering = list(reversed(list(nx.topological_sort(self.G))))
        print "Topological ordering: ", topological_ordering
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
                print "Popped function: ", fn_name
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
        print "Visit function: ", node.name
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
                num_iter = self.eval_args_helper(node.iter.args[0]) #node.iter.args[0].n
            elif len(node.iter.args) == 2:
                # Start and an end
                num_iter = self.eval_args_helper(node.iter.args[1]) - self.eval_args_helper(node.iter.args[0])#node.iter.args[1].n - node.iter.args[0].n 
            else:
                # Start, end and a step
                num_iter = (self.eval_args_helper(node.iter.args[1]) - self.eval_args_helper(node.iter.args[0])) / self.eval_args_helper(node.iter.args[2])#(node.iter.args[1].n - node.iter.args[0].n) / node.iter.args[2].n
        except Exception as e:
            return 


        self.for_to_iters[node.lineno] = num_iter
        if node.lineno in self.fns_to_calls.keys():
            self.counter += self.fns_to_calls[node.lineno]
            return
        
        before_visit = self.counter
        print "In for loop: ", node.lineno
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

        print "For loop {0} calls the target function {1} times".format(node.lineno, diff * num_iter)


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
            print "Calling function: ", fn_name
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


class ConstantPropagation(ast.NodeTransformer):
    """NodeTransformer that will inline any Number and String 
    constants defined on the module level wherever they are used
    throughout the module. Any Number or String variable matching [A-Z_]+ 
    on the module level will be used as a constant"""

    def __init__(self):
        self._constants = {}
        super(ConstantPropagation, self).__init__()

    def visit_Module(self, node):
        """Find eglible variables to be inlined and store
        the Name->value mapping in self._constants for later use"""
        assigns = [x for x in node.body if type(x) == ast.Assign]
        for assign in assigns:
            if type(assign.value) in (ast.Num, ast.Str):
                for name in assign.targets:
                    print "name: ", name.id
                    self._constants[name.id] = assign.value
        return self.generic_visit(node)

    def visit_Name(self, node):
        """If node.id is in self._constants, replace the
        loading of the node with the actual value"""
        
        #return self._constants.get(node.id, node)
        for k in self._constants.keys():
            if node.id == k:
                return self._constants[k]

        return node


    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        return node


class ASTChecks(ast.NodeTransformer):
    def visit_If(self, node):
        raise ValueError("Currently, control flow logic like if/else is not supported. Please use alternative conditionals like array_index_secret_if")


class ASTParser(object):
    
    def __init__(self, fname, debug=False):
        f = open(fname, 'r')
        s = f.read()
        f.close()
        if mpc_type == SPDZ:
            s = "open_channel(0)\n" + s + "\nclose_channel(0)\n"
        self.tree = ast.parse(s)

        self.debug = debug

    def parse(self):
        # Run through a bunch of parsers
        # hardcoded to count the # of matmul calls. Returns the # of matmul calls.
        target = "matmul"
        # Try inlining
        self.tree = ConstantPropagation().visit(self.tree)
        dep = ProcessDependencies()
        dep.visit(self.tree)
        count_calls = CountFnCall(dep.G, dep.functions, target)
        count_calls.visit(self.tree)
        count_calls.postprocess()

        print "Number of matmul calls: ", count_calls.counter
        print "Functions to calls: ", count_calls.fns_to_calls
        print "Matmul to dimensions: ", count_calls.matmul_to_dims
        print "Matmul to calls: ", count_calls.matmul_to_calls
        print "Vectorized Triples requirement: ", count_calls.vectorized_calls

        self.tree = ForLoopParser().visit(self.tree)
        self.tree = ASTChecks().visit(self.tree)

        return count_calls.vectorized_calls

    def execute(self, context):
        source = astor.to_source(self.tree)
        if self.debug:
            print(source)
        exec(source, context)
        #exec(compile(self.tree, filename="<ast>", mode="exec"), context)
