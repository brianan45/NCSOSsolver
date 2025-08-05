import sympy as sp
from sympy.matrices.expressions import MatAdd, MatMul

def is_zero_matrix_term(expr):
    """Return True if expr is structurally zero."""
    # Pure scalar zero
    if expr == 0 or expr is sp.S.Zero:
        return True
    # Multiplication containing a scalar zero
    if isinstance(expr, sp.Mul) and any(f == 0 or f is sp.S.Zero for f in expr.args):
        return True
    return False

def remove_zero_terms(expr):
    """Recursively remove 0*... terms from matrix expressions."""
    
    if isinstance(expr, MatAdd):
        cleaned_args = [remove_zero_terms(arg) for arg in expr.args]
        cleaned_args = [arg for arg in cleaned_args if not is_zero_matrix_term(arg)]
        return sp.S.Zero if not cleaned_args else MatAdd(*cleaned_args)
    
    elif isinstance(expr, MatMul):
        if any(is_zero_matrix_term(arg) for arg in expr.args):
            return sp.S.Zero
        cleaned_args = [remove_zero_terms(arg) for arg in expr.args]
        return MatMul(*cleaned_args)
    
    elif isinstance(expr, sp.Pow):
        return sp.Pow(remove_zero_terms(expr.base), expr.exp)
    
    elif isinstance(expr, sp.Adjoint):
        return sp.Adjoint(remove_zero_terms(expr.arg))
    
    else:
        return expr