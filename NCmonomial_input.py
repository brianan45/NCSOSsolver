import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor
)
import re

def preprocess_monomials(expr_str):
    # Convert ^T to .T and insert * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # Convert X^* to X, since we assume Hermitian matrices
    expr_str = re.sub(r'(\w+)\^\*', r'\1', expr_str)
    return expr_str

def insert_multiplication(expr_str, var_names):
    # Ensure variables like A0, B0, etc. are not split (match longest first)
    var_names = sorted(var_names, key=len, reverse=True)
    pattern = '|'.join(re.escape(name) for name in var_names)

    # Tokenize: split where variables meet (e.g., A0B0 → A0 * B0)
    tokens = re.findall(pattern, expr_str)
    return '*'.join(tokens)

def get_vars_vec():
    var_input = input("Enter the matrix variable names (assumed to be Hermitian) (comma-separated): ")
    # print("var_input =", var_input)
    var_names = [name.strip() for name in var_input.split(',')]
    if "I" not in var_names:
        var_names.append("I")
    # print("var_names =", var_names)

    n = sp.Symbol('n', integer=True, positive=True)
    matrices = {name: sp.MatrixSymbol(name, n, n) for name in var_names}
    vars = {matrices[name] for name in var_names}

    monomial_input = input("Enter the monomial expressions (comma-separated; enter ^T for transpose, ^* for conjugate transpose, AB for A*B): ")
    raw_monomials = [expr.strip() for expr in monomial_input.split(',')]
    # print("raw_monomials =", raw_monomials)

    processed_monomials = [
        insert_multiplication(preprocess_monomials(expr), var_names)
        for expr in raw_monomials
    ]
    # print("processed_monomials =", processed_monomials)

    # Local dictionary for SymPy parsing
    local_dict = matrices.copy()
    local_dict['Adjoint'] = sp.Adjoint
    local_dict['I'] = sp.Identity(n)  # Allow identity matrix as 'I'

    transformations = standard_transformations + (convert_xor,)

    try:
        monomial_exprs = [
            parse_expr(expr, local_dict=local_dict, transformations=transformations, evaluate=False)
            for expr in processed_monomials
        ]
        # print("monomial_exprs =", monomial_exprs)
        monomial_vector = sp.Matrix(monomial_exprs)
        return n, vars, monomial_vector
    except Exception as e:
        print("Error parsing monomial expressions:", e)
        return n, vars, matrices