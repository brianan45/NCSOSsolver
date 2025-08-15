import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor
import itertools
import re

# a function that gets the variables and monomials from user input

def insert_multiplication(expr, var_names):
    # Example: turn 'AB' into 'A*B'
    for name in var_names:
        expr = expr.replace(name, f"{name}*")
    # Remove the last '*' if present
    if expr.endswith('*'):
        expr = expr[:-1]
    return expr

def preprocess_monomials(expr_str):
    # Convert ^T to .T and insert * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # Convert X^* to X, since we assume Hermitian matrices
    expr_str = re.sub(r'(\w+)\^\*', r'\1', expr_str)
    return expr_str

def generate_monomials(var_names, degree):
    # if user inputs a number n instead of a monomial vector, generate a
    # vector of all possible monomials in the given variables, up to degree n
    """Generate all monomials (as strings) up to given degree."""
    monomials = ["I"]  # Start with identity
    # Exclude I from variable list for generation
    variables = [v for v in var_names if v != "I"]

    for d in range(1, degree + 1):
        for prod in itertools.product(variables, repeat=d):
            monomials.append("".join(prod))
    return monomials

def get_vars_vec():
    var_input = input("Enter the matrix variable names (assumed to be Hermitian) (comma-separated): ")
    var_names = [name.strip() for name in var_input.split(',')]
    if "I" not in var_names:
        var_names.append("I")

    n = sp.Symbol('n', integer=True, positive=True)
    matrices = {name: sp.MatrixSymbol(name, n, n) for name in var_names}
    vars = {matrices[name] for name in var_names}

    monomial_input = input("Enter the monomial expressions (comma-separated; OR single integer degree for auto-generation): ").strip()

    # If input is a single integer, auto-generate monomials
    if monomial_input.isdigit():
        degree = int(monomial_input)
        raw_monomials = generate_monomials(var_names, degree)
    else:
        raw_monomials = [expr.strip() for expr in monomial_input.split(',')]

    processed_monomials = [
        insert_multiplication(preprocess_monomials(expr), var_names)
        for expr in raw_monomials
    ]

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
        monomial_vector = sp.Matrix(monomial_exprs)
        return n, vars, monomial_vector
    except Exception as e:
        print("Error parsing monomial expressions:", e)
        return n, vars, matrices
