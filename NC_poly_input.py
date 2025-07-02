import sympy as sp
import re
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

def preprocess_monomials(expr_str):
    """
    Preprocesses input to handle matrix powers, adjoints, transposes, etc.
    """
    # Convert ^T to .T and add * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # Convert X^* to Adjoint(X)
    expr_str = re.sub(r'(\w+)\^\*', r'Adjoint(\1)', expr_str)

    # Insert * between Adjoint(...) and following symbol
    expr_str = re.sub(r'(Adjoint\(\w+\))(?=\w)', r'\1*', expr_str)

    # Convert X^3 to X**3
    expr_str = re.sub(r'(\w+)\^(\d+)', r'\1**\2', expr_str)

    # Final adjoint check
    expr_str = re.sub(r'(Adjoint\(\w\))(?=\w)', r'\1*', expr_str)

    print("expr_str =", expr_str)
    return expr_str

def get_poly(vars, n):
    """
    Prompts user to input a matrix-valued polynomial expression and parses it.
    Automatically maps I to Identity(n).
    """
    var_dict = {var.name: sp.MatrixSymbol(var.name, n, n) for var in vars}
    var_dict["Adjoint"] = sp.Adjoint
    var_dict["I"] = sp.Identity(n)  # ✅ Add identity matrix symbol

    print("var_dict =", var_dict)

    poly_input = input(f"Enter a polynomial in terms of {', '.join(str(v) for v in vars)}: ")
    poly_input = preprocess_monomials(poly_input)
    print("type(poly_input) =", type(poly_input))

    try:
        polynomial = parse_expr(
            poly_input,
            local_dict=var_dict,
            transformations=transformations,
            evaluate=False
        )
        if not isinstance(polynomial, sp.Expr):
            print("Warning: Input is not a valid expression.")
        return polynomial
    except Exception as e:
        print("Error parsing polynomial:", e)
        return None
