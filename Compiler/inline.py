import ast

from .fat_tools import (OptimizerStep, NodeTransformer, NodeVisitor,
                    pretty_dump, get_starargs, get_keywords, get_varkeywords, copy_node)


import copy

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
                    print "Parent: {0}, name: {1}".format(parent, name)
                    print self.fns_to_vars[parent]
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
        fn_def_node = copy.deepcopy(node)
        self.scope_stack.insert(0, fn_def_node.name)
        args = []
        for arg in fn_def_node.args.args:
            # Parameters are exclusive to that scope so must rename them. No checks needed.
            if isinstance(arg, ast.Name):
                arg.id = self.remap(node.name, arg.id)
                args.append(arg.id)


        self.fns_to_params[fn_def_node.name] = args

        fn_def_node = self.generic_visit(fn_def_node)
        self.fn_to_node[fn_def_node.name] = fn_def_node
        self.scope_stack.pop(0)
        return fn_def_node

    def visit_Call(self, node):
        call_node = copy.deepcopy(node)
        parent = self.get_parent()
        if isinstance(call_node.func, ast.Attribute):
            if parent:
                call_node.func.value.id = self.remap(parent, call_node.func.value.id)
            self.visit(call_node.func)
        else:
            pass
            #print "Calling function: ", node.func.id

        # Check arguments passed into a function call.
        for i in range(len(call_node.args)):
            #self.visit(arg)
            arg = call_node.args[i]
            if isinstance(arg, ast.Name):
                name = arg.id
                if parent and name in self.fns_to_vars[parent]:
                    remapped_name = self.remap(parent, arg.id)
                    call_node.args[i] = ast.Name(id=remapped_name)

        return call_node

    def visit_Assign(self, node):
        parent = self.get_parent()
        assign_node = copy.deepcopy(node)
        if isinstance(assign_node.targets[0], ast.Name):
            name = assign_node.targets[0].id  
            if parent:           
                assign_node.targets[0].id = self.remap(parent, name)
            
        elif parent and isinstance(assign_node.targets[0], ast.Tuple):
            for i in range(len(assign_node.targets[0].elts)):
                name = assign_node.targets[0].elts[i].id
                if parent:
                    assign_node.targets[0].elts[i].id = self.remap(parent, name)
                else:
                    self.global_names.add(name)
        elif parent and isinstance(assign_node.targets[0], ast.Subscript):
            assign_node.targets[0] = self.visit(assign_node.targets[0])
            print assign_node.targets[0].__dict__
            
        if isinstance(assign_node.value, ast.Name):
            if parent and assign_node.value.id not in self.global_names:
                assign_node.value.id = self.remap(parent, assign_node.value.id)

        assign_node.value = self.visit(assign_node.value)
        return assign_node



    def visit_BinOp(self, node):
        parent = self.get_parent()
        copy_binop = copy.deepcopy(node)
        self.resolve_binop_names(parent, copy_binop)
        return copy_binop

    def visit_Subscript(self, node):
        parent = self.get_parent()
        copy_node = copy.deepcopy(node)
        subscript_obj = copy_node
        while isinstance(subscript_obj, ast.Subscript):
            subscript_obj.slice = self.visit(subscript_obj.slice)
            subscript_obj = subscript_obj.value

        if parent and subscript_obj.id not in self.global_names:
            subscript_obj.id = self.remap(parent, subscript_obj.id)

        return copy_node


    def visit_Index(self, node):
        print "Visit index: ", node.__dict__
        parent = self.get_parent()
        index_node = copy.deepcopy(node)
        if isinstance(index_node.value, ast.Name):
            if parent and index_node.value.id not in self.global_names:
                index_node.value.id = self.remap(parent, index_node.value.id)

        elif isinstance(index_node.value, ast.BinOp):
            self.resolve_binop_names(parent, index_node.value)

        return index_node


    def visit_Return(self, node):
        parent = self.get_parent()
        return_node = copy.deepcopy(node)
        if parent and isinstance(return_node.value, ast.Name): #and node.value.id not in self.global_names :
            return_node.value.id = self.remap(parent, return_node.value.id)
        elif parent and isinstance(return_node.value, ast.Tuple):
            for name_obj in return_node.value.elts:
                #if name_obj.id not in self.global_names:
                name_obj.id = self.remap(parent, name_obj.id)

        return return_node

    def resolve_binop_names(self, parent, binop):
        if isinstance(binop, ast.Name):
            binop.id = self.remap(parent, binop.id)
            return  

        if isinstance(binop.left, ast.Subscript):
            binop.left = self.visit(binop.left)


        if isinstance(binop.right, ast.Subscript):
            binop.right = self.visit(binop.right)

        if isinstance(binop.left, ast.Name):
            binop.left.id = self.remap(parent, binop.left.id)

        if isinstance(binop.right, ast.Name):
            binop.right.id = self.remap(parent, binop.right.id)

        try:
            left = self.resolve_binop_names(parent, binop.left)
            right = self.resolve_binop_names(parent, binop.right)
            if isinstance(left, ast.Name) and left.id not in self.global_names:
                left.id = self.remap(parent, left.id)

            if isinstance(right, ast.Name) and right.id not in self.global_names:
                right.id = self.remap(parent, right.id)
        except Exception as e:
            print "Resolve Binop Names Exception: ", e 



    # TODO: Multiassignment of for.
    def visit_For(self, node):
        for_node = ast.For(target=node.target, iter=node.iter, body=node.body, orelse=node.orelse)
        parent = self.get_parent()
        # Rename the variable, so for i in range, track the i
        if isinstance(for_node.target, ast.Name):
            if parent:
                self.fns_to_vars[parent].add(for_node.target.id)
                for_node.target.id = self.remap(parent, for_node.target.id)
                #print "FOR LOOP HERE PARENT:", parent
        elif isinstance(for_node.target, ast.Tuple):
            for name_obj in for_node.target.elts:
                if parent:
                    self.fns_to_vars[parent].add(name_obj.id)
                    name_obj.id = self.remap(parent, name_obj.id)


        # Rename the variable in 'range', for i in range(x), rename the variable x as well
        for_node.iter = self.visit(for_node.iter)
        for_node = self.generic_visit(for_node)
        """
        for item in for_node.body:
            self.visit(item)
        """
        return for_node

    """
    def copy_node(self, node):
        if isinstance(node, ast.Name):
            return ast.Name(id=node.id)
        elif isinstance(node, ast.Num):
            return ast.Num(n=node.n)
        elif isinstance(node, ast.For):
            return ast.For(target=node.target, iter=node.iter, body=node.body, orelse=node.orelse)
        elif isinstance(node, ast.Assign):
            return ast.Assign(targets=node.targets, value=node.value)
        elif isinstance(node, ast.Index):
            return ast.Index(value=node.value)
        elif isinstance(node, ast.Call):
            return ast.Call(func=node.func, args=node.args, keywords=node.keywords)
        elif isinstance(node, ast.Subscript):
            return ast.Subscript(slice=node.slice, value=node.value)
        else:
            print "ERROR: Copy type not found: ", type(node), node.__dict__
            return node
    """
    


import networkx as nx 
import astunparse
import copy
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
        self.fns_to_returns[parent] = copy.deepcopy(node.value)
        return copy.deepcopy(node)


    def add_assignments(self, args, params):
        lst_assign = []
        for arg, param in zip(args, params):
            if arg != None:
                print "Add assignment of arg: {0} to param: {1}".format(arg,param)
                if isinstance(arg, str):
                    assign_obj = ast.Assign(targets=[ast.Name(id=param)], value=ast.Name(id=arg))
                    lst_assign.append(assign_obj)
                elif type(arg) in (int, float):
                    assign_obj = ast.Assign(targets=[ast.Name(id=param)], value=ast.Num(n=arg))
                    lst_assign.append(assign_obj)
                else:
                    pass 
                

        return lst_assign

    def visit_Assign(self, node):
        parent = self.get_parent()
        if isinstance(node.value, ast.Call):
            retval = self.visit_Call_helper(node.value)
            if isinstance(retval, list):
                print "Inline assignment objs:"
                for assignment_obj in retval + [node]:
                    try:
                        print assignment_obj.targets[0].id
                    except Exception as e:
                        pass

                # Assign return values to original call values.
                return_value = self.fns_to_returns[node.value.func.id]
                print "Inline return: ", return_value
                if isinstance(return_value, ast.Tuple):
                    lst_assign = []
                    for i in range(len(node.targets[0].elts)):
                        ast_assign_node = ast.Assign(targets=[node.targets[0].elts[i]], value=return_value.elts[i])
                        print "Inline Assign: ", node.targets[0].elts[i], return_value.elts[i]
                        lst_assign.append(ast_assign_node)

                    new_assignment_node = lst_assign
                    retval = retval + new_assignment_node
                else:
                    new_assignment_node = ast.Assign(targets=node.targets, value=return_value)
                    retval = retval + [new_assignment_node]
                print "INLINE RETURN VALUE: ", return_value
                print "Assignment: ", new_assignment_node
                
                retval = [copy.deepcopy(ele) for ele in retval]
                return retval
            else:
                return copy.deepcopy(node)
            
        return copy.deepcopy(node)

    def visit_Call_helper(self, node):
        parent = self.get_parent()
        try:
            fn_name = node.func.id
            if fn_name in self.fns_to_params.keys(): 
                print "fn_name: ", fn_name, self.fns_to_params.keys()
                params = self.fns_to_params[fn_name]
                fn_def_node = self.fns_to_defs[fn_name]
                args = []
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        args.append(arg.id)
                    elif isinstance(arg, ast.Num):
                        args.append(arg.n)
                    else:
                        args.append(None)
                lst_assign = self.add_assignments(args, params)
                if isinstance(fn_def_node.body[-1], ast.Return):
                    retval = lst_assign + fn_def_node.body[:-1]
                else:
                    retval = lst_assign + fn_def_node.body

                retval = [copy.deepcopy(ele) for ele in retval]
                return retval
    
        except AttributeError as e:
            print "INLINE EXCEPTION:", e
            return copy.deepcopy(node)


        return copy.deepcopy(node) 

    def visit_Call(self, node):
        inline_call_node = self.visit_Call_helper(node)
        if not isinstance(inline_call_node, list):
            return inline_call_node
        elif isinstance(inline_call_node[-1], ast.Return):
            return inline_call_node[:-1]
        else:
            return inline_call_node