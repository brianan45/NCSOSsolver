import re

def process_monomials(expr_str):
    """
    Preprocesses input to handle matrix powers, adjoints, transposes, etc.
    """
    # Convert ^T to .T and add * if followed by a symbol
    expr_str = expr_str.replace('^T', '.T')
    expr_str = re.sub(r'\.T(?=\w)', '.T*', expr_str)

    # # Convert X^* to Adjoint(X)
    # expr_str = re.sub(r'(\w+)\^\*', r'Adjoint(\1)', expr_str)

    # Convert X^* to X, since we assume Hermitian matrices
    expr_str = re.sub(r'(\w+)\^\*', r'\1', expr_str)

    # # Insert * between Adjoint(...) and following symbol
    # expr_str = re.sub(r'(Adjoint\(\w+\))(?=\w)', r'\1*', expr_str)

    # Convert X^3 to X**3
    expr_str = re.sub(r'(\w+)\^(\d+)', r'\1**\2', expr_str)

    # Final adjoint check
    # expr_str = re.sub(r'(Adjoint\(\w\))(?=\w)', r'\1*', expr_str)

    # print("expr_str =", expr_str)
    return expr_str