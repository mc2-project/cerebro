import ast

from .fat_tools import (OptimizerStep, NodeTransformer, NodeVisitor,
                    pretty_dump, get_starargs, get_keywords, get_varkeywords)

class Checker(ast.NodeVisitor):
    '''Gather a list of problems that would prevent inlining a function.'''
    def __init__(self):
        self.problems = []

    def visit_Call(self, node):
        # Reject explicit attempts to use locals()
        # FIXME: detect uses via other names
        if isinstance(node.func, ast.Name):
            if node.func.id == 'locals':
                self.problems.append('use of locals()')


def locate_kwarg(funcdef, name):
    '''Get the index of an argument of funcdef by name.'''
    for idx, arg in enumerate(funcdef.args.args):
        if arg.arg == name:
            return idx
    raise ValueError('argument %r not found' % name)



"""
class RenameVisitor(ast.NodeTransformer):
    # FIXME: Reuse tools.ReplaceVariable

    def __init__(self, callsite, inlinable, actual_pos_args):
        #assert get_starargs(callsite) is None
        #assert not get_varkeywords(callsite) is not None
        #assert inlinable.args.vararg is None
        #assert inlinable.args.kwonlyargs == []
        #assert inlinable.args.kw_defaults == []
        #assert inlinable.args.kwarg is None
        #assert inlinable.args.defaults == []

        # Mapping from name in callee to node in caller
        self.remapping = {}
        for formal, actual in zip(inlinable.args.args, actual_pos_args):
            self.remapping[formal.id] = actual

    def visit_Name(self, node):
        if node.id in self.remapping:
            return self.remapping[node.id]
        return node
"""



class RenameVisitorHelper(ast.NodeVisitor):
    def __init__(self, fns_to_params):
        self.fns_to_vars = {}
        for fn in fns_to_params.keys():
            self.fns_to_vars[fn] = set()


        self.scope_stack = []



    def get_parent(self):
        if len(self.scope_stack):
            return self.scope_stack[0]
        else:
            return None

    def visit_FunctionDef(self, node):
        self.scope_stack.insert(0, node.name)
        for arg in node.args.args:
            # Arguments are exclusive to that scope.
            if isinstance(arg, ast.Name):
                self.fns_to_vars[node.name].add(arg.id)

        self.generic_visit(node)
        self.scope_stack.pop(0)


    def visit_Assign(self, node):
        parent = self.get_parent()
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id  
            if parent:           
                self.fns_to_vars[parent].add(name)
            
        elif parent and isinstance(node.targets[0], ast.Tuple):
            for i in range(len(node.targets[0].elts)):
                name = node.targets[0].elts[i].id
                if parent:
                    self.fns_to_vars[parent].add(name)


# Renames the parameters and all corresponding references within the function. This helps SO much with later creating assignment statements.
class RenameVisitor(ast.NodeTransformer):
    def __init__(self, fns_to_params, fns_to_vars):
        self.fns_to_params = fns_to_params
        self.remapping = {}
        for fn_name in self.fns_to_params.keys():
            for param in self.fns_to_params[fn_name]:
                self.remapping[(fn_name, param)] = fn_name + "_" + param
        
        self.scope_stack = []  
        self.global_names = set()
        # Hardcoded
        self.hardcoded_names = ["sfix", "cfix", "sint", "s_fix", "c_fix", "sfixMatrix", "s_fix_mat", "cfixMatrix", "sfixMatrixGC"]

        self.fns_to_vars = fns_to_vars
        self.fn_to_node = {}



    def remap(self, parent, name, for_loop=False):
        if name not in self.hardcoded_names:
            if name.startswith(parent):
                return name 
            else:
                if name not in self.fns_to_vars[parent] and not for_loop:
                    # Not a variable exclusive to that scope
                    return name
                else:
                    return parent + "_" + name
        else:
            return name

    def get_parent(self):
        if len(self.scope_stack):
            return self.scope_stack[0]
        else:
            return None

    def visit_FunctionDef(self, node):
        self.scope_stack.insert(0, node.name)
        args = []
        for arg in node.args.args:
            # Parameters are exclusive to that scope so must rename them. No checks needed.
            if isinstance(arg, ast.Name):
                arg.id = self.remap(node.name, arg.id)
                args.append(arg.id)


        self.fns_to_params[node.name] = args

        node = self.generic_visit(node)
        self.fn_to_node[node.name] = node
        self.scope_stack.pop(0)
        return node

    def visit_Call(self, node):
        parent = self.get_parent()
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func)
        else:
            pass
            #print "Calling function: ", node.func.id

        # Check arguments passed into a function call.
        for arg in node.args:
            #self.visit(arg)
            if isinstance(arg, ast.Name):
                name = arg.id
                if parent and name in self.fns_to_vars[parent]:
                    arg.id = self.remap(parent, arg.id)

        return node

    def visit_Assign(self, node):
        parent = self.get_parent()
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id  
            if parent:           
                node.targets[0].id = self.remap(parent, name)
            
        elif parent and isinstance(node.targets[0], ast.Tuple):
            for i in range(len(node.targets[0].elts)):
                name = node.targets[0].elts[i].id
                if parent:
                    node.targets[0].elts[i].id = self.remap(parent, name)
                else:
                    self.global_names.add(name)
        elif parent and isinstance(node.targets[0], ast.Subscript):
            self.visit(node.targets[0])
            print node.targets[0].__dict__
            
        if isinstance(node.value, ast.Name):
            if parent and node.value.id not in self.global_names:
                node.value.id = self.remap(parent, node.value.id)

        self.visit(node.value)
        return node



    def visit_Subscript(self, node):
        parent = self.get_parent()
        subscript_obj = node
        while isinstance(subscript_obj, ast.Subscript):
            self.visit(subscript_obj.slice)
            subscript_obj = subscript_obj.value

        if subscript_obj.id not in self.global_names:
            subscript_obj.id = self.remap(parent, subscript_obj.id)

        return node


    def visit_Index(self, node):
        parent = self.get_parent()
        if isinstance(node.value, ast.Name):
            if node.value.id not in self.global_names:
                node.value.id = self.remap(parent, node.value.id)

        elif isinstance(node.value, ast.BinOp):
            self.resolve_binop_names(parent, node.value)

        return node


    def visit_Return(self, node):
        parent = self.get_parent()
        print "Return", node.__dict__
        if isinstance(node.value, ast.Name): #and node.value.id not in self.global_names :
            node.value.id = self.remap(parent, node.value.id)
        elif isinstance(node.value, ast.Tuple):
            for name_obj in node.value.elts:
                #if name_obj.id not in self.global_names:
                name_obj.id = self.remap(parent, name_obj.id)

        return node

    def resolve_binop_names(self, parent, binop):
        if isinstance(binop, ast.Name):
            binop.id = self.remap(parent, binop.id)
            return  
        left = self.resolve_binop_names(parent, binop.left)
        right = self.resolve_binop_names(parent, binop.right)
        if isinstance(left, ast.Name) and left.id not in self.global_names:
            self.remap(parent, left.id)

        if isinstance(right, ast.Name) and right.id not in self.global_names:
            self.remap(parent, right.id)


    # TODO: Multiassignment of for.
    def visit_For(self, node):
        parent = self.get_parent()
        # Rename the variable, so for i in range, track the i
        if isinstance(node.target, ast.Name):
            self.fns_to_vars[parent].add(node.target.id)
            node.target.id = self.remap(parent, node.target.id)
            print "FOR LOOP HERE:", node.target.id
        elif isinstance(node.target, ast.Tuple):
            for name_obj in node.target.elts:
                self.fns_to_vars[parent].add(name_obj.id)
                name_obj.id = self.remap(parent, name_obj.id)


        # Rename the variable in 'range', for i in range(x), rename the variable x as well
        self.visit(node.iter)

        for item in node.body:
            self.visit(item)
        return node

    # Need to remap ALL variables, excluding global variables
    """
    def visit_Name(self, node):
        parent = self.get_parent()
        if parent and (parent, node.id) in self.remapping.keys() and (parent, node.id) not in self.seen:
            node.id = self.remap(parent, node.id)
            self.seen.add((parent, node.id))

        return node
    """

import networkx as nx 
import astunparse
class Expansion:
    '''Information about a callsite that's a candidate for inlining, giving
    the funcdef, and the actual positional arguments (having
    resolved any keyword arguments.'''
    def __init__(self, funcdef, actual_pos_args):
        self.funcdef = funcdef
        self.actual_pos_args = actual_pos_args



class InlineSubstitution(OptimizerStep, ast.NodeTransformer):
    """Function call inlining."""

    def __init__(self, fns_to_defs={}, fns_to_params={}, dep_graph=None):
        self.fns_to_defs = fns_to_defs
        self.fns_to_params = fns_to_params
        self.scope_stack = []
        self.fn_def_stack = []
        self.G = dep_graph
        self.fns_to_returns = {}

        self.preprocess()
        


    # Get the new fns to params
    def preprocess(self):
        print list(nx.algorithms.dag.topological_sort(self.G))
        for node in reversed(list(nx.algorithms.dag.topological_sort(self.G))):
            if node in self.fns_to_defs.keys():
                fn_def_node = self.fns_to_defs[node]
                self.visit(fn_def_node)



    def get_parent(self):
        if len(self.scope_stack):
            return self.scope_stack[0]
        else:
            return None

    
    def visit_FunctionDef(self, node):
        parent = self.get_parent()
        self.scope_stack.insert(0, node.name)
        self.fn_def_stack.insert(0, node)
        self.generic_visit(node)
        self.scope_stack.pop(0)
        node = self.fn_def_stack.pop(0)
        self.fns_to_defs[node.name] = node
        return node



    def visit_Return(self, node):
        parent = self.get_parent()
        self.fns_to_returns[parent] = node.value
        return node


    def add_assignments(self, args, params):
        lst_assign = []
        for arg, param in zip(args, params):
            print "Add assignment of arg: {0} to param: {1}".format(arg,param)
            assign_obj = ast.Assign(targets=[ast.Name(id=param)], value=ast.Name(id=arg))
            lst_assign.append(assign_obj)

        return lst_assign

    def visit_Assign(self, node):
        parent = self.get_parent()
        if isinstance(node.value, ast.Call):
            #call_node =  self.visit(node.value)

            #if isinstance(node, list):
                #return node
            #else:
                #return node
            retval = self.visit_Call_helper(node.value)
            if isinstance(retval, list):
                print "AHAHAAA"
                for assignment_obj in retval + [node]:
                    try:
                        print assignment_obj.targets[0].id
                    except Exception as e:
                        pass


                # Assign return values to original call values.
                new_assignment_node = ast.Assign(targets=node.targets, value=self.fns_to_returns[node.value.func.id])
                return retval + [new_assignment_node]
            else:
                return node
            
        return node

    def visit_Call_helper(self, node):
        parent = self.get_parent()
        try:
            fn_name = node.func.id
            print "fn_name: ", fn_name, self.fns_to_params.keys()
            if fn_name in self.fns_to_params.keys(): 
                params = self.fns_to_params[fn_name]
                fn_def_node = self.fns_to_defs[fn_name]
                args = [arg.id for arg in node.args]
                lst_assign = self.add_assignments(args, params)
                #func = self.fns_to_defs[fn_name]
                #func.body = lst_assign + func.body
                if isinstance(fn_def_node.body[-1], ast.Return):
                    return lst_assign + fn_def_node.body[:-1]
                else:
                    return lst_assign + fn_def_node.body
    
        except AttributeError as e:
            print e
            return node


        return node 




    def visit_Call(self, node):
        inline_call_node = self.visit_Call_helper(node)
        if not isinstance(inline_call_node, list):
            return inline_call_node
        elif isinstance(inline_call_node[-1], ast.Return):
            return inline_call_node[:-1]
        else:
            return inline_call_node
