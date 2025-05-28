import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

# a function that gets a set of variables and a monomial vector from input

def get_monomial_vector():
    var_input = input("Enter the variables (comma-separated): ")
    variables = sp.symbols(var_input)
    if isinstance(variables, sp.Symbol):
        variables = (variables,)

    monomial_input = input(
        f"Enter the monomials in terms of {', '.join(str(v) for v in variables)} (comma-separated with natural number exponents): ")

    try:
        local_dict = {str(v): v for v in variables}
        transformations = (standard_transformations +
                           (implicit_multiplication_application, convert_xor))
        monomials = [parse_expr(m.strip(), local_dict=local_dict,
                                transformations=transformations)
                     for m in monomial_input.split(',')]
        monomial_vector = sp.Matrix(monomials)
        return monomial_vector, variables
    except Exception as e:
        print("Error parsing input:", e)
        return sp.Matrix([]), []
