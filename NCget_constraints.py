import sympy as sp
import re
from process_monomials import process_monomials
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)

def get_constraints(vars, n):
    # var_dict = {str(v): v for v in vars}
    constraints = []

    var_dict = {var.name: sp.MatrixSymbol(var.name, n, n) for var in vars}
    var_dict["Adjoint"] = sp.Adjoint
    var_dict["I"] = sp.Identity(n)  # ✅ Add identity matrix symbol

    while True:
        inp = input(f"Enter >= 0 constraints g({', '.join(str(v) for v in vars)}) (comma-separated) (press Enter to finish): ").strip()
        if inp == "":
            break

        # Split by comma and preprocess each expression
        exprs = inp.split(',')

        # for expr_str in exprs:
        #     expr_str = expr_str.strip()
        #     expr_str = expr_str.replace("^", "**")
        #     expr_str = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', expr_str)

        for expr_str in exprs:
            expr_str = process_monomials(expr_str)

            try:
                # g = sp.sympify(expr, locals=var_dict)
                expr = parse_expr(
                    expr_str,
                    local_dict=var_dict,
                    transformations=transformations,
                    evaluate=False
                )
                constraints.append(expr)
            except Exception as e:
                print(f"Could not parse constraint '{expr_str}': {e}")
    return constraints