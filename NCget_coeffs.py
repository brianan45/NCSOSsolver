import sympy as sp
from sympy import Add, Mul
from collections import defaultdict

def get_coeffs(expr1, expr2, vars):
    """
    Solve for unknown coefficients by matching noncommutative terms in two expressions.
    
    Parameters:
        expr1, expr2: SymPy expressions with NC variables (e.g., a*x + b*x*y + 1)
        vars: list of NC variables (e.g., [x, y])
    
    Returns:
        Dictionary of solved coefficients or None if inconsistent.
    """
    def extract_terms(expr):
        expr = sp.expand(expr)
        result = defaultdict(lambda: 0)
        if isinstance(expr, Add):
            terms = expr.args
        else:
            terms = [expr]

        for term in terms:
            if isinstance(term, Mul):
                coeff, *rest = term.as_ordered_factors()
                # separate coefficient from variables
                coeff_part = 1
                mon_part = []
                for f in term.args:
                    if any(v in f.free_symbols for v in vars):
                        mon_part.append(f)
                    else:
                        coeff_part *= f
                monomial = sp.Mul(*mon_part, evaluate=False)
                result[monomial] += coeff_part
            else:
                if any(v in term.free_symbols for v in vars):
                    result[term] += 1
                else:
                    result[1] += term
        return result

    dict1 = extract_terms(expr1)
    dict2 = extract_terms(expr2)

    all_monomials = set(dict1.keys()) | set(dict2.keys())

    eqns = [sp.Eq(dict1.get(m, 0), dict2.get(m, 0)) for m in all_monomials]

    sol = sp.solve(eqns)

    if not sol:
        return None
    return sol