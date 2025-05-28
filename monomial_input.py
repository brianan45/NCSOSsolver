import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def get_monomial_vector():
    # Step 1: Get variables from the user
    var_input = input("Enter the variables (comma-separated): ")
    variables = sp.symbols(var_input)
    if isinstance(variables, sp.Symbol):
        variables = (variables,)  # Make it a tuple

    # Step 2: Get the monomials from the user
    monomial_input = input(f"Enter the monomials in terms of {', '.join([str(v) for v in variables])} (comma-separated): ")

    # Step 3: Parse the monomials into SymPy expressions with implicit multiplication
    try:
        local_dict = {str(v): v for v in variables}
        transformations = standard_transformations + (implicit_multiplication_application,)
        monomials = [parse_expr(m.strip(), local_dict=local_dict, transformations=transformations)
                     for m in monomial_input.split(',')]
        monomial_vector = sp.Matrix(monomials)  # Make it a column vector
        return monomial_vector, variables
    except Exception as e:
        print("Error parsing input:", e)
        return sp.Matrix([]), []
