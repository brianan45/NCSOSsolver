import sympy as sp
from sympy.matrices.expressions import MatrixSymbol
import re

def add_mult(expr_str: str, matrices: dict[str, MatrixSymbol]) -> sp.MatrixExpr:
    """
    Parses a string like "A0A1 - B0B1" into a valid SymPy matrix expression,
    inserting * (matrix multiplication) between adjacent matrix variables.

    Parameters:
        expr_str (str): Expression as a string.
        matrices (dict): Dictionary mapping variable names (like 'A0') to MatrixSymbol objects.

    Returns:
        MatrixExpr: Parsed SymPy matrix expression.
    """
    # Sort variable names by length descending to avoid partial matches (e.g., A vs A0)
    sorted_names = sorted(matrices.keys(), key=len, reverse=True)

    # Tokenize the input string by identifying known matrix variable names
    pattern = '|'.join(re.escape(name) for name in sorted_names)
    tokens = re.findall(pattern + r'|\*\*|\*|\+|\-|\(|\)|\d+|[^\s]', expr_str)

    # Insert '*' between adjacent matrix symbols or between matrix symbols and parentheses
    result_tokens = []
    prev_token = None
    for token in tokens:
        if prev_token:
            prev_is_var = prev_token in matrices
            curr_is_var = token in matrices
            curr_is_lparen = token == '('
            # If two variables are adjacent or var followed by '(', insert *
            if (prev_is_var and (curr_is_var or curr_is_lparen)):
                result_tokens.append('*')
        result_tokens.append(token)
        prev_token = token

    # Join into an expression string
    joined = ' '.join(result_tokens)

    # Replace variable names with matrices["name"]
    for name in sorted_names:
        joined = re.sub(rf'\b{name}\b', f'matrices["{name}"]', joined)

    # Evaluate safely
    return eval(joined, {"matrices": matrices, "sp": sp})