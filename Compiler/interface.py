from types import *
from types_gc import *
import compilerLib, library

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

class ASTChecks(ast.NodeTransformer):
    def __init__(self):
        self.if_stack = []
        self.test_name = "test"
        self.test_counter = 0

    def merge_test_single(self, test):
        if test[1]:
            return ast.Name(id=test[0], ctx=ast.Load())
        else: 
            return ast.BinOp(left=ast.Num(1), op=ast.Sub(), right=ast.Name(id=test[0], ctx=ast.Load()))

    def merge_test_list(self, test_list):
        ret = test_list[0]
        ret = self.merge_test_single(ret)
        for test in test_list[1:]:
            ret = ast.BinOp(left=ret, op=ast.Mult(), right=self.merge_test_single(test))

        return ret

    def assign_transform(self, target, test_list, a, b):
        test_v = self.merge_test_list(test_list)
        left_sum = ast.BinOp(left=test_v, op=ast.Mult(), right=a)
        test_neg = ast.BinOp(left=ast.Num(1), op=ast.Sub(), right=test_v)
        right_sum = ast.BinOp(left=test_neg, op=ast.Mult(), right=b)
        return ast.Assign(targets=[target], value=ast.BinOp(left=left_sum, op=ast.Add(), right=right_sum))
    
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

        print "Statements: ", test_name, statements

        self.if_stack.append((test_name, True))
        for n in node.body:
            if isinstance(n, ast.If):
                statements += self.visit(n)
            elif isinstance(n, ast.Assign):
                s = self.assign_transform(n.targets[0], self.if_stack, n.value, n.targets[0])
                if not isinstance(s, list):
                    statements.append(s)
                else:
                    statements += s
            else:
                raise ValueError("Does not support non-assignment statments within if statements")
        self.if_stack.pop()

        self.if_stack.append((test_name, False))
        for n in node.orelse:
            if isinstance(n, ast.If):
                statements += self.visit(n)
            elif isinstance(n, ast.Assign):
                s = self.assign_transform(n.targets[0], self.if_stack, n.value, n.targets[0])
                if not isinstance(s, list):
                    statements.append(s)
                else:
                    statements += s
            else:
                raise ValueError("Does not support non-assignment statments within if statements")
        self.if_stack.pop()
        
        ast.copy_location(statements[0], node)
        counter = 0
        for s in statements:
            s.lineno = statements[0].lineno + counter
            s.col_offset = statements[0].col_offset
            counter += 1

        return statements

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
        self.tree = ForLoopParser().visit(self.tree)
        self.tree = ASTChecks().visit(self.tree)

    def execute(self, context):
        source = astor.to_source(self.tree)
        if self.debug:
            print(source)
        exec(source, context)
        #exec(compile(self.tree, filename="<ast>", mode="exec"), context)
