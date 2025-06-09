import sympy as sp
import re
from monomial_input import get_monomial_vector

def preprocess_input(expr_str):
    # Replace ^ with **
    expr_str = expr_str.replace('^', '**')

    # Replace standalone 'i' with 'I' (imaginary unit)
    expr_str = re.sub(r'(?<![\w])i(?![\w])', 'I', expr_str)

    # Insert * between number/variable and parenthesis or variable: e.g., 2(x+y) → 2*(x+y), x2y → x*2*y
    expr_str = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', expr_str)

    # Insert * between I (imaginary unit) and numbers/variables/parentheses, e.g., 3Ix → 3*I*x
    expr_str = re.sub(r'(?<=[^*])I(?=[a-zA-Z0-9(])', 'I*', expr_str)

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