import sympy as sp
import re
from monomial_input import get_monomial_vector

def preprocess_input(expr_str):
    # Step 1: Replace ^ with ** for exponentiation
    expr_str = expr_str.replace('^', '**')

    # Step 2: Replace all instances of 'i' with 'I' (imaginary unit)
    expr_str = expr_str.replace('i', 'I')

    # Step 3: Insert multiplication sign * where needed (between number/letter and letter/paren)
    expr_str = re.sub(r'(?<=[a-zA-Z0-9)])(?=[(a-zA-Z])', '*', expr_str)
    
    return expr_str

def get_polynomial_from_input(variables):
    """
    Prompts the user to input a polynomial expression and parses it using sympy.

    Args:
        variables (tuple): Tuple of sympy.Symbol variables (e.g., (x, y, z))

    Returns:
        sympy.Expr: Parsed polynomial expression
    """
    var_dict = {str(v): v for v in variables}
    # Also include 'I' in locals so sympy can evaluate it properly
    var_dict['I'] = sp.I

    poly_input = input(f"Enter a polynomial in terms of {', '.join(var_dict.keys())} (use 'i' for √-1): ")
    poly_input = preprocess_input(poly_input)

    try:
        polynomial = sp.sympify(poly_input, locals=var_dict)
        if not polynomial.is_polynomial(*variables):
            print("Warning: Input is not a valid polynomial.")
        return polynomial
    except Exception as e:
        print("Error parsing polynomial:", e)
        return None