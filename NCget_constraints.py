import sympy as sp
import re
from process_monomials import process_monomials
from add_mult import add_mult
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

transformations = standard_transformations + (convert_xor,)

# transformations = standard_transformations + (convert_xor,)

def get_constraints(vars, n):
    # var_dict = {str(v): v for v in vars}
    constraints = []
    constraintsI = []
    # print([var.name for var in vars])
    var_dict = {var.name: sp.MatrixSymbol(var.name, n, n) for var in vars}
    print("type(var.name) =", type(next(iter(var_dict.keys()))))
    print("var_dict =", var_dict)
    print("type(var_dict(values)) =", type(next(iter(var_dict.values()))))
    # for name, symbol in var_dict.items():
    #     print(f"{name}: is_commutative = {symbol.is_commutative}")
    var_dict["Adjoint"] = sp.Adjoint
    var_dict["I"] = sp.Identity(n)  # ✅ Add identity matrix symbol


    inp = input(f"Enter >= 0 constraints g({', '.join(str(v) for v in vars)}) (comma-separated) (press Enter to finish): ").strip()

    # Split by comma and preprocess each expression
    exprs = inp.split(',')

    for expr_str in exprs:
        expr_str = process_monomials(expr_str)
        print(f"Processing constraint: {expr_str}")

        try:
        #     # g = sp.sympify(expr, locals=var_dict)
        #     expr = parse_expr(
        #         expr_str,
        #         local_dict=var_dict,
        #         transformations=transformations,
        #         evaluate=True
        #     )
            expr = add_mult(expr_str, var_dict)
            print(f"Parsed constraint: {expr}")

            constraints.append(expr)
        except Exception as e:
            print(f"Could not parse constraint '{expr_str}': {e}")

    inp = input(f"Enter = I constraints g({', '.join(str(v) for v in vars)}) (comma-separated) (press Enter to finish): ").strip()

    # Split by comma and preprocess each expression
    exprs = inp.split(',')

    for expr_str in exprs:
        expr_str = process_monomials(expr_str)

        try:
            expr = parse_expr(
                expr_str,
                local_dict=var_dict,
                transformations=transformations,
                evaluate=True
            )
            constraintsI.append(expr)
        except Exception as e:
            print(f"Could not parse constraint '{expr_str}': {e}")
    return constraints, constraintsI
