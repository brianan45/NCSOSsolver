import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
import re

def preprocess_monomials(expr_str):
    # Convert ^T to .T and add * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # Convert every `X^*` into `Adjoint(X)`
    expr_str = re.sub(r'(\w+)\^\*', r'Adjoint(\1)', expr_str)

    # Insert * between Adjoint(...) and following variable, if needed
    expr_str = re.sub(r'(Adjoint\(\w+\))(?=\w)', r'\1*', expr_str)
    return expr_str

def get_vars_vec():
    var_input = input("Enter the matrix variable names (comma-separated): ")
    var_names = [name.strip() for name in var_input.split(',')]

    n = sp.Symbol('n', integer=True, positive=True)
    matrices = {name: sp.MatrixSymbol(name, n, n) for name in var_names}
    vars = {sp.MatrixSymbol(name, n, n) for name in var_names}

    monomial_input = input("Enter the monomial expressions (comma-separated; enter ^T for transpose, ^* for conjugate transpose, AB for A*B): ")
    raw_monomials = [expr.strip() for expr in monomial_input.split(',')]
    processed_monomials = [preprocess_monomials(expr) for expr in raw_monomials]

    local_dict = matrices.copy()
    local_dict['Adjoint'] = sp.Adjoint
    local_dict['I'] = sp.Identity(n)  # Add the identity matrix as 'I'

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor
    )

    try:
        monomial_exprs = [
            parse_expr(expr, local_dict=local_dict, transformations=transformations)
            for expr in processed_monomials
        ]
        monomial_vector = sp.Matrix(monomial_exprs)
        return n, vars, monomial_vector
    except Exception as e:
        print("Error parsing monomial expressions:", e)
        return n, vars, matrices
