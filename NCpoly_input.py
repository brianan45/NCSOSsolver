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

# a function that takes a polynomial from user input

def preprocess_monomials(expr_str):
    """
    Preprocesses input to handle matrix powers, adjoints, transposes, etc.
    """
    # Convert ^T to .T and add * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # Convert X^* to X, since we assume Hermitian matrices
    expr_str = re.sub(r'(\w+)\^\*', r'\1', expr_str)

    # Replace ^ with **
    expr_str = expr_str.replace('^', '**')

    # Insert * between variables/numbers and variables/parentheses (e.g. x2y → x*2*y)
    expr_str = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', expr_str)

    return expr_str

def get_poly(variables, n):
    """
    Prompts the user to input a polynomial expression and parses it using sympy.

    Args:
        variables (tuple): Tuple of sympy.Symbol variables (e.g., (x, y, z))

    Returns:
        sympy.Expr: Parsed polynomial expression
    """
    var_dict = {var.name: sp.MatrixSymbol(var.name, n, n) for var in variables}
    var_dict["Adjoint"] = sp.Adjoint
    var_dict["I"] = sp.Identity(n)  # ✅ Add identity matrix symbol
    poly_input = input(f"Enter a polynomial in terms of {', '.join(var_dict.keys())}: ")
    poly_input = preprocess_monomials(poly_input)

    try:
        polynomial = sp.sympify(poly_input, locals=var_dict)
        return polynomial
    except Exception as e:
        print("Error parsing polynomial:", e)
        return None