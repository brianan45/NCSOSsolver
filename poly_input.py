# polynomial_input.py

import sympy as sp

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

    try:
        polynomial = sp.sympify(poly_input, locals=var_dict)
        if not sp.Poly(polynomial, *variables).is_polynomial():
            print("Warning: Input is not a valid polynomial.")
        return polynomial
    except Exception as e:
        print("Error parsing polynomial:", e)
        return None
