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

class SecretFixedPointArrayFactory(object):
    def __call__(self, length):
        if mpc_type == SPDZ:
            return sfixArray(length)
        else:
            return sfixArrayGC(length)

    @classmethod
    def read_input(self, length, party):
        if mpc_type == SPDZ:
            ret = sfixArray(length)
            @for_range(ret.length)
            def f(i):
                v = sint.get_private_input_from(party)
                ret[i] = sfix.load_sint(v)
            return ret
        else:
            ret = sfixArrayGC(length)
            for i in range(ret.length):
                v = sint_gc(Params.intp, party)
                ret[i] = sfix_gc.load_sint(v)
            return ret


class SecretFixedPointMatrixFactory(object):
    def __call__(self, rows, columns):
        if mpc_type == SPDZ:
            return sfixMatrix(rows, columns)
        else:
            return sfixMatrixGC(rows, columns)

    @classmethod
    def read_input(self, rows, columns, party):
        if mpc_type == SPDZ:
            ret = sfixMatrix(rows, columns)
            @for_range(ret.rows)
            def f(i):
                @for_range(ret.columns)
                def g(j):
                    v = sint.get_private_input_from(party)
                    ret[i][j] = sfix.load_sint(v)
            return ret
        else:
            ret = sfixMatrixGC(rows, columns)
            for i in range(ret.rows):
                for j in range(ret.columns):
                    v = sint_gc(Params.intp, party)
                    ret[i][j] = sfix_gc.load_sint(v)
            return ret

ClearInteger = ClearIntegerFactory()
SecretInteger = SecretIntegerFactory()
SecretFixedPoint = SecretFixedPointFactory()
SecretFixedPointArray = SecretFixedPointArrayFactory()
SecretFixedPointMatrix = SecretFixedPointMatrixFactory()

compilerLib.VARS["c_int"] = ClearInteger
compilerLib.VARS["s_int"] = SecretInteger
compilerLib.VARS["s_fix"] = SecretFixedPoint
compilerLib.VARS["s_fix_array"] = SecretFixedPointArray
compilerLib.VARS["s_fix_mat"] = SecretFixedPointMatrix

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

    def visit_If(self, node):
        raise ValueError("Currently, control flow logic like if/else is not supported. Please use alternative conditionals like array_index_secret_if")
