from types import *
from types_gc import *
import compilerLib, library

SPDZ = 0
GC = 1

class Params(object):
    intp = 32
    f = 32
    k = 32

    @classmethod
    def set_params(cls, int_precision=32, f=32, k=64):
        cls.intp = int_precision
        cls.f = f
        cls.k = k
        sfix.set_precision(f, k)
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

class SecretFixedPointFactory(object):
    def __call__(self, party):
        if mpc_type == SPDZ:
            return sint.get_private_input_from(party)
        else:
            return sint_gc(Params.intp, party)
    
    def read_input(self, party):
        if mpc_type == SPDZ:
            v = sint.get_private_input_from(party)
            vf = sfix.load_sint(v)
            return vf
        else:
            return sfix_gc(v=None, input_party=party)

class SecretFixedPointMatrixFactory(object):
    def __call__(self, rows, columns):
        if mpc_type == SPDZ:
            return sfixMatrix(rows, columns)
        else:
            return sfixMatrixGC(rows, columns)

def forloop(start, stop=None, step=None):
    def decorator(func):
        if mpc_type == SPDZ:
            return library.for_range(start, stop, step)(func)
        else:
            if stop is None:
                for i in range(start):
                    func(i)
            elif step is None:
                for i in range(start, stop):
                    func(i)
            else:
                for i in range(start, stop, step):
                    func(i)
    return decorator
    

ClearInteger = ClearIntegerFactory()
SecretInteger = SecretIntegerFactory()
SecretFixedPoint = SecretFixedPointFactory()
SecretFixedPointMatrix = SecretFixedPointMatrixFactory()

compilerLib.VARS["c_int"] = ClearInteger
compilerLib.VARS["s_int"] = SecretInteger
compilerLib.VARS["s_fix"] = SecretFixedPoint
compilerLib.VARS["s_fix_mat"] = SecretFixedPointMatrix
compilerLib.VARS["forloop"] = forloop

import ast
import astor
from pprint import pprint

class ASTParser(ast.NodeTransformer):
    
    def __init__(self, fname, debug=False):
        f = open(fname, 'r')
        self.tree = ast.parse(f.read())
        f.close()
        self.forloop_counter = 0
        self.debug = debug

    def parse(self):
        self.visit(self.tree)

    def execute(self, context):
        source = astor.to_source(self.tree)
        if self.debug:
            print(source)
        exec(compile(self.tree, filename="<ast>", mode="exec"), context)
        
    def visit_For(self, node):
        self.generic_visit(node)

        dec = ast.Call(func=ast.Name(id="forloop", ctx=ast.Load()), args=node.iter.args, keywords=[])
        new_node = ast.FunctionDef(name="for{}".format(self.forloop_counter),
                                   args=ast.arguments(args=[node.target], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                                   body=node.body,
                                   keywords=[],
                                   decorator_list=[dec])
        self.forloop_counter += 1
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        
        return new_node
