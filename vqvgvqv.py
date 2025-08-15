import sympy as sp
from sympy import Mul
from NCget_coeffs import get_coeffs
from sub_identity_matrix import sub_identity_matrix, get_matrix_exps

# A function that extracts the coefficients of the equation vQv + (sum of (vQv * g * vQv)) - p - lm * I = 0

def vqvgvqv(p, v, matrix_vars, Q0_sym, g_list, I_list, Qi_syms):
    def reverse_monovec(expr_matrix):
        result = expr_matrix.copy()
        for i in range(expr_matrix.rows):
            entry = expr_matrix[i, 0]
            if isinstance(entry, Mul):
                reversed_entry = Mul(*reversed(entry.args), evaluate=False)
                result[i, 0] = reversed_entry
        return result
    

    sum_vQv = ((reverse_monovec(v)).T * Q0_sym * v)[0, 0]
    for g_sym, Qi_sym in zip(g_list, Qi_syms):
        sum_vQv += ((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym  * ((reverse_monovec(v)).T * Qi_sym * v)[0, 0]

    sum_vQv = sp.expand(sum_vQv)

    matrix_exps = get_matrix_exps(I_list)
    print("matrix_exps =", matrix_exps)
    simplified = sub_identity_matrix(sub_identity_matrix(sp.expand(sum_vQv - p), matrix_exps),matrix_exps)
    print("sum_vQv - p =", sp.expand(sum_vQv - p))
    print("simplified =", simplified)

    sol = get_coeffs(simplified, matrix_vars)
    return sol