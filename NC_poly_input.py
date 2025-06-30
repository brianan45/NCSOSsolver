import sympy as sp
import re
from monomial_input import get_monomial_vector

# a function that, given a set of variables, accepts a
# polynomial in those variables from user input

def preprocess_monomials(expr_str):
    # Convert ^T to .T and add * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # 1. Convert every `X^*` into `Adjoint(X)`
    expr_str = re.sub(r'(\w+)\^\*', r'Adjoint(\1)', expr_str)
    
    # 2. Insert * between Adjoint(...) and following variable, if needed
    expr_str = re.sub(r'(Adjoint\(\w+\))(?=\w)', r'\1*', expr_str)
    
    # Convert exponentiation like X^3 into X**3
    expr_str = re.sub(r'(\w+)\^(\d+)', r'\1**\2', expr_str)

    # If Adjoint(...) is followed immediately by a symbol/letter, add a *
    expr_str = re.sub(r'(Adjoint\(\w\))(?=\w)', r'\1*', expr_str)
    print("expr_str =", expr_str)
    return expr_str

def get_poly(vars, n):
    """
    Prompts the user to input a polynomial expression and parses it using sympy.

    Args:
        variables (tuple): Tuple of sympy.Symbol variables (e.g., (x, y, z))

    Returns:
        sympy.Expr: Parsed polynomial expression
    """
    var_dict = {var.name: sp.MatrixSymbol(var.name, n, n) for var in vars}
    print("var_dict =", var_dict)
    # print("type(variables) =", type(variables))
    poly_input = input(f"Enter a polynomial in terms of {', '.join(str(v) for v in vars)}: ")
    poly_input = preprocess_monomials(poly_input)

    try:
        polynomial = sp.sympify(poly_input, locals=var_dict)
        print("type(polynomial) =", type(polynomial))
        if not polynomial.is_polynomial(*var_dict):
            print("Warning: Input is not a valid polynomial.")
        return polynomial
    except Exception as e:
        print("Error parsing polynomial:", e)
        return None
