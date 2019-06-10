from types import *
from types_gc import *
import compilerLib, library
import ast
import symtable
import re
import networkx as nx

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
    print isinstance(v, Matrix)
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

class ForLoopParser(ast.NodeTransformer):
    num_matmul = 0
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
    def __init__(self):

        # map function name to function definition.
        self.functions = {}
        self.scope_stack = []
        # If encounter target function call, instantly add 1
        # Functions are nodes, dependencies are edges
        self.G = nx.DiGraph()


    def visit_FunctionDef(self, node):
        # Not considering differing scopes and 
        self.functions[node.name] = node

        if node.name not in self.G.nodes:
            self.G.add_node(node.name)

        print "Visiting function: ", node.name
        if len(self.scope_stack):
            "Function ", node.name, " was called by ", self.scope_stack[0]

        # Enter a function, put it on the stack
        self.scope_stack.insert(0, node.name)
        # Visit the children in the function
        self.generic_visit(node)
        self.scope_stack.pop(0)


    def visit_Call(self, node):
        if not len(self.scope_stack):
            parent = None
        else:
            parent = self.scope_stack[0]

        # Some nodes don't have ids?
        if 'id' in node.func.__dict__.keys():
            print "Function {0} called by {1}".format(node.func.id, parent)
            self.scope_stack.insert(0, node.func.id)
            print "First time visiting: ", node.func.id
            self.generic_visit(node)
            if parent:
                self.G.add_edge(parent, node.func.id)

            self.scope_stack.pop(0)

    def visit_For(self, node):
        # Parse the for loop to get the # of iterations. Won't work if variable # of iterations so try to avoid that.
        print "For loop lineno: ", node.lineno
        self.functions[node.lineno] = node
        if len(self.scope_stack):
            print "Parent:", self.scope_stack[0]


        if node.lineno not in self.G.nodes:
            self.G.add_node(node.lineno)

        if len(self.scope_stack):
            print "For loop: Adding edge from: {0} to {1}".format(self.scope_stack[0], node.lineno)
            self.G.add_edge(self.scope_stack[0], node.lineno)

        self.scope_stack.insert(0, node.lineno)
        #print "Visiting for loop in function: ", self.scope_stack[0]
        for item in node.body:
            self.visit(item)

        self.scope_stack.pop(0)

    def visit_Assign(self, node):
        self.visit(node.value)


class CountFnCall(ast.NodeVisitor):
    def __init__(self, G, functions, target_fn="matmul"):
        self.functions = functions 
        self.fns_to_calls = {}
        self.G = G
        self.target_fn = target_fn
        self.counter = 0
        # Contains ONLY the functions that are defined. Previously had to add in scopes for for-loops in self.functions. AH bad names.
        self.only_functions = set()
        self.only_functions.add(self.target_fn)

        self.fns_to_calls[target_fn] = 1
        # Gather all the necessary information for counting calls.
        self.process()


    def process(self):
        topological_ordering = list(reversed(list(nx.topological_sort(self.G))))
        print "Topological ordering: ", topological_ordering
        for node in topological_ordering:
            if node in self.functions.keys():
                self.visit(self.functions[node])


        # Reset the counter
        self.counter = 0


    def visit_FunctionDef(self, node):
        print "Visit function: ", node.name
        before_visit = self.counter 
        self.generic_visit(node)
        after_visit = self.counter
        diff = after_visit - before_visit
        if node.name not in self.fns_to_calls.keys():
            self.fns_to_calls[node.name] = diff

        self.only_functions.add(node.name)
        self.counter = before_visit

    def visit_For(self, node):
        try:
            if len(node.iter.args) == 1:
                num_iter = node.iter.args[0].n
            elif len(node.iter.args) == 2:
                num_iter = node.iter.args[1].n - node.iter.args[0].n 
            else:
                num_iter = (node.iter.args[1].n - node.iter.args[0].n) / node.iter.args[2].n
        except Exception as e:
            return 

        
        if node.lineno in self.fns_to_calls.keys():
            self.counter += self.fns_to_calls[node.lineno]
            return
        
        before_visit = self.counter
        print "In for loop: ", node.lineno
        for item in node.body:
            if isinstance(item, ast.For) and item.lineno in self.fns_to_calls.keys():
                self.counter += self.fns_to_calls[item.lineno]
            else: 
                self.visit(item)

        after_visit = self.counter
        diff = after_visit - before_visit
        if node.lineno not in self.fns_to_calls.keys():
            self.fns_to_calls[node.lineno] = diff * num_iter

        print "For loop {0} calls the target function {1} times".format(node.lineno, diff * num_iter)

    def visit_Call(self, node):
        before_visit = self.counter
        if 'id' in node.func.__dict__.keys():
            print "Calling function: ", node.func.id
            if node.func.id not in self.fns_to_calls.keys():
                self.generic_visit(node)
                after_visit = self.counter 
                diff = after_visit - before_visit
                self.fns_to_calls[node.func.id] = diff 

            self.counter = before_visit + self.fns_to_calls[node.func.id]
            print "Function {0} calls the target_fn {1} times.".format(node.func.id, self.fns_to_calls[node.func.id])

    def visit_Assign(self, node):
        self.visit(node.value)



    def subtract_initial(self):
        res = self.counter
        for k in self.fns_to_calls.keys():
            pass
            #if k not in self.only_functions:
                # Subtract away the initial definitions that occurred when I initially parsed the file.
                #res -= self.fns_to_calls[k]

        return res

class ConstantMaker(ast.NodeTransformer):
    """NodeTransformer that will inline any Number and String 
    constants defined on the module level wherever they are used
    throughout the module. Any Number or String variable matching [A-Z_]+ 
    on the module level will be used as a constant"""

    def __init__(self):
        self._constants = {}
        super(ConstantMaker, self).__init__()

    def visit_Module(self, node):
        """Find eglible variables to be inlined and store
        the Name->value mapping in self._constants for later use"""

        assigns = [x for x in node.body if type(x) == ast.Assign]

        for assign in assigns:
            if type(assign.value) in (ast.Num, ast.Str):
                for name in assign.targets:
                    self._constants[name] = assign.value

        return self.generic_visit(node)

    def visit_Name(self, node):
        """If node.id is in self._constants, replace the
        loading of the node with the actual value"""
        
        #return self._constants.get(node.id, node)
        for k in self._constants.keys():
            if node.lineno == k.lineno:
                return node
            elif node.id == k.id:
                return self._constants[k]

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
        # hardcoded to count the # of matmul calls.
        target = "matmul"
        self.tree = ConstantMaker().visit(self.tree)
        dep = ProcessDependencies()
        dep.visit(self.tree)
        print dep.G.edges()
        print dep.G.nodes()
        count_calls = CountFnCall(dep.G, dep.functions, target)
        count_calls.visit(self.tree)
        #process.visit(self.tree)
        print "Number of matmul calls before: ", count_calls.counter
        counter = count_calls.subtract_initial()

        
        print "Number of matmul calls: ", counter
        print "Functions to calls: ", count_calls.fns_to_calls
        self.tree = ForLoopParser().visit(self.tree)
        self.tree = ASTChecks().visit(self.tree)

    def execute(self, context):
        source = astor.to_source(self.tree)
        if self.debug:
            print(source)
        exec(source, context)
        #exec(compile(self.tree, filename="<ast>", mode="exec"), context)
