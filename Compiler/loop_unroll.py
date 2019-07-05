import ast

from .fat_tools import (OptimizerStep, ReplaceVariable, FindNodes, NodeTransformer,
                    compact_dump, copy_lineno,
                    ITERABLE_TYPES)


CANNOT_UNROLL = (ast.Break, ast.Continue, ast.Raise)


#class UnrollStep(OptimizerStep, ast.NodeTransformer):
class UnrollStep(OptimizerStep, ast.NodeTransformer):
    def _visit_For(self, node):
        """
        if not isinstance(node.target, ast.Name):
            return

        # for i in (1, 2, 3): ...
        if not isinstance(node.iter, ast.Num):
            return node 
        iter_value = node.iter.value
        if not isinstance(iter_value, tuple):
            print "HEY", iter_value
            return node
        #if not(1 <= len(iter_value) <= self.config.unroll_loops):
            #return
        """
        #found = None
        # don't optimize if 'break' or 'continue' is found in the loop body
        """
        def find_callback(node):
            nonlocal found
            found = node
            return False
        """
        # FIXME: restrict this the current scope
        # (don't enter class/function def/list comprehension/...)
        """
        visitor = FindNodes(CANNOT_UNROLL, find_callback)
        visitor.visit(node)
        if found is not None:
            self.log(node,
                     "cannot unroll loop: %s is used at line %s",
                     compact_dump(found),
                     found.lineno)
            return
        """

        try:
            if len(node.iter.args) == 1:
                # Ex: for i in range(6)
                #num_iter = self.eval_args_helper(node.iter.args[0])
                lst_iters = list(range(self.eval_args_helper(node.iter.args[0])))
            elif len(node.iter.args) == 2:
                # Start and an end
                # Ex: for i in range(6, 8)
                #num_iter = self.eval_args_helper(node.iter.args[1]) - self.eval_args_helper(node.iter.args[0])
                lst_iters = list(range(self.eval_args_helper(node.iter.args[0]), self.eval_args_helper(node.iter.args[1])))
            else:
                # Start, end and a step
                # Ex: for i in range(6, 12, 2)
                #num_iter = (self.eval_args_helper(node.iter.args[1]) - self.eval_args_helper(node.iter.args[0])) / self.eval_args_helper(node.iter.args[2])
                lst_iters = list(range(self.eval_args_helper(node.iter.args[0]), self.eval_args_helper(node.iter.args[1]),  self.eval_args_helper(node.iter.args[2])))
        except Exception as e:
            print "EXCEPTION", e
            return node



        name = node.target.id
        body = node.body
        print "Loop var name", name
        # replace 'for i in (1, 2, 3): body' with...
        new_node = []
        #for value in node.iter.value:
        for value in lst_iters:
            #value_ast = _new_constant(node.iter, value) #self._new_constant(node.iter, value)
            #print "Value ast: ", value_ast
            value_ast = ast.Num(n=value)
            if value_ast is None:
                return node

            # 'i = 1'
            name_ast = ast.Name(id=name, ctx=ast.Store())
            copy_lineno(node, name_ast)
            assign = ast.Assign(targets=[name_ast],
                                value=value_ast)
            copy_lineno(node, assign)
            new_node.append(assign)

            # duplicate 'body'
            for item in body:
                if isinstance(item, ast.For):
                    new_node.extend(self.visit(item))
                else:
                    new_node.append(item)
            #new_node.extend(body)

        if node.orelse:
            new_node.extend(node.orelse)

        #self.log(node, "unroll loop (%s iterations)", len(node.iter.value))
        return new_node

    def visit_For(self, node):
        #if not self.config.unroll_loops:
            #return

        new_node = self._visit_For(node)
        if new_node is None:
            return node

        # loop was unrolled: run again the optimize on the new nodes
        #return self.visit_node_list(new_node)
        return new_node



    def eval_args_helper(self, node):
        if hasattr(node, 'n'):
            return node.n
        else:
            left_val = self.eval_args_helper(node.left)
            right_val = self.eval_args_helper(node.right)
            res = operators[type(node.op)](left_val, right_val)
            return res


class UnrollListComp:
    def unroll_comprehension(self, node):
        if not self.config.unroll_loops:
            return

        # FIXME: support multiple generators
        # [i for i in range(3) for y in range(3)]
        if len(node.generators) > 1:
            return

        generator = node.generators[0]
        if not isinstance(generator, ast.comprehension):
            return
        # FIXME: support if
        if generator.ifs:
            return

        if not isinstance(generator.target, ast.Name):
            return
        target = generator.target.id

        if not isinstance(generator.iter, ast.Constant):
            return
        iter_value = generator.iter.value
        if not isinstance(iter_value, ITERABLE_TYPES):
            return
        if not(1 <= len(iter_value) <= self.config.unroll_loops):
            return

        if isinstance(node, ast.DictComp):
            keys = []
            values = []
            for value in iter_value:
                ast_value = self.new_constant(node, value)
                if ast_value is None:
                    return
                replace = ReplaceVariable(self.filename, {target: ast_value})

                key = replace.visit(node.key)
                keys.append(key)

                value = replace.visit(node.value)
                values.append(value)

            new_node = ast.Dict(keys=keys, values=values, ctx=ast.Load())
        else:
            items = []
            for value in iter_value:
                ast_value = self.new_constant(node, value)
                if ast_value is None:
                    return
                replace = ReplaceVariable(self.filename, {target: ast_value})
                item = replace.visit(node.elt)
                items.append(item)

            # FIXME: move below?
            if isinstance(node, ast.SetComp):
                new_node = ast.Set(elts=items, ctx=ast.Load())
            else:
                assert isinstance(node, ast.ListComp)
                new_node = ast.List(elts=items, ctx=ast.Load())

        copy_lineno(node, new_node)
        return new_node
