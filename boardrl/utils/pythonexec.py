import ast
import io
from contextlib import redirect_stdout, redirect_stderr
from collections import namedtuple

ExecOutput = namedtuple("ExecOutput", ["retval", "stdout", "stderr"])


class PythonExec:
    def __init__(self, code):
        self.ctx = {}
        block = ast.parse(code, mode="exec")

        # Handle empty code case
        last_expr = None
        if block.body:
            last_stmt = block.body[-1]
            # Check if the last statement is an expression
            if isinstance(last_stmt, ast.Expr):
                # Extract the expression and remove from exec block
                last_expr_node = block.body.pop()
                assert isinstance(last_expr_node, ast.Expr)
                last_expr = ast.Expression(last_expr_node.value)

        self.exec_compiled = compile(block, "<string>", "exec")
        self.eval_compiled = None
        if last_expr is not None:
            self.eval_compiled = compile(last_expr, "<string>", "eval")

    def __call__(self, vars):
        self.ctx.update(vars)
        retval = None
        exec(self.exec_compiled, self.ctx, self.ctx)
        if self.eval_compiled is not None:
            retval = eval(self.eval_compiled, self.ctx, self.ctx)
        return retval
