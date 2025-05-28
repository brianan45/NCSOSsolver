import sympy as sp
import re
from monomial_input import get_monomial_vector

def preprocess_input(expr_str):
    # Replace ^ with **
    expr_str = expr_str.replace('^', '**')
    # Insert * between variables/numbers and variables/parentheses (e.g. x2y → x*2*y)
    expr_str = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', expr_str)
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
    poly_input = input(f"Enter a polynomial in terms of {', '.join(var_dict.keys())}: ")
    poly_input = preprocess_input(poly_input)

    try:
        polynomial = sp.sympify(poly_input, locals=var_dict)
        if not polynomial.is_polynomial(*variables):
            print("Warning: Input is not a valid polynomial.")
        return polynomial
    except Exception as e:
        print("Error parsing polynomial:", e)
        return None