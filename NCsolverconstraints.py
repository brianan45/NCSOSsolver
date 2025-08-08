import sympy as sp
from sympy.logic.boolalg import BooleanFalse
from sympy import Mul, S
import cvxpy as cp
import numpy as np
import sys
from NCmonomial_input import get_vars_vec
from NCpoly_input import get_poly
from sp_to_cp_constraint_multiQ import sp_to_cp_constraint_multiQ
from NCget_coeffs import get_coeffs
from NCget_constraints import get_constraints
from matrix_sqrt import matrix_sqrt
from clean_value import clean_value
from sub_identity_matrix import sub_identity_matrix, get_matrix_exps
from vqvgvqv import vqvgvqv
from remove_zero_terms import remove_zero_terms

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

n, matrix_vars, v = get_vars_vec()
m = v.shape[0]
lm_sp = sp.Symbol("lm_sp")
p = get_poly(matrix_vars, n)
p = -p
p += lm_sp * sp.Identity(n)
g_list, I_list = get_constraints(matrix_vars, n)
# Since g_list is a list of constraints g(x) = 0, we need to negate them
# to convert them to the form g(x) >= 0 for the SOS solver, since we require
# nonnegatvity constraints and g = 0 implies -g >= 0 and g >= 0
g_list.extend([-g for g in g_list])

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q0 = cp.Variable((m, m), PSD=True)  # for main SOS
Qi_list = [cp.Variable((m, m), PSD=True) for _ in g_list]  # one per constraint
lm_cp = cp.Variable(name="lm_cp")

# Build the expression v^T Q v symbolically
Q0_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'q0_{min(i,j)}_{max(i,j)}'))
Qi_syms = [
    sp.Matrix(m, m, lambda i, j, k=k: sp.Symbol(f'q{k+1}_{min(i,j)}_{max(i,j)}'))
    for k in range(len(g_list))
]

# A function that takes a vector whose entries are matrices or matrix products and reverses
# the order of matrix multiplication (ex.: v = [AB, CD] → reverse_monovec(v) = [BA, DC])
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
    sum_vQv += ((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym  # * ((reverse_monovec(v)).T * Qi_sym * v)[0, 0]

sum_vQv = sp.expand(sum_vQv)  

# Get a dictionary of matrix variables and their exponents that yield the
# identity matrix (ex.: I_list = [A^2] → get_matrix_exps(I_list) = {A: 2})
matrix_exps = get_matrix_exps(I_list)
# Substitute the identity matrix wherever possible
# We apply substitutions to the expression sum_vQv - p, which equals 0
simplified = sub_identity_matrix(sub_identity_matrix(sp.expand(sum_vQv - p), matrix_exps),matrix_exps)

# Extract the coefficients of the polynomial and set all to 0
sol = get_coeffs(simplified, matrix_vars)

# Check for False anywhere in the solution list
if any(isinstance(s, BooleanFalse) for s in sol):
    sol = vqvgvqv(p, v, matrix_vars, Q0_sym, g_list, I_list, Qi_syms)
    if any(isinstance(s, BooleanFalse) for s in sol):
        print("Problem is infeasible. Exiting.")
        sys.exit()

extra_subs = {lm_sp: lm_cp}

cvx_eqs = []

# Replace the SymPy variables in sol with their original cvxpy variables
for eq in sol:
    cvx_eqs.append(sp_to_cp_constraint_multiQ(eq, [Q0] + Qi_list, extra_subs=extra_subs))

cvx_eqs.append(lm_cp >= 0)

problem = cp.Problem(cp.Minimize(lm_cp),cvx_eqs)
problem.solve(solver=cp.SCS)

if problem.status == "infeasible":
    print("Problem is infeasible. Exiting.")
    sys.exit()


print("\nLambda:")
print(clean_value(lm_cp.value))

print("Q0 matrix =", clean_value(Q0.value))
Q0_sqrt = sp.Matrix(matrix_sqrt(clean_value(Q0.value)))
vQ0v_sqrt = Q0_sqrt.T @ v
vQ0v = (clean_value(sp.Adjoint(vQ0v_sqrt) * vQ0v_sqrt)[0,0])

SOS_decomp = vQ0v

# for i, g in enumerate(g_list):
#     if Qi_list[i].value is not None:
#         print("Qi matrix:", clean_value(Qi_list[i].value))
#         Qi_sqrt = sp.Matrix(matrix_sqrt(Qi_list[i].value))
#         vQv_sqrt = (Qi_sqrt.T @ v)
#         vQv = (clean_value(sp.Adjoint(vQv_sqrt) * vQv_sqrt)[0,0])
#         SOS_decomp = sp.Add(SOS_decomp, sp.MatMul(vQv, g, evaluate=False), evaluate=False)
SOS_decomp = remove_zero_terms(SOS_decomp)
print("\nSOS decomposition:", str(SOS_decomp).replace("Adjoint", "Adj"))

# WORKING CHSH EXAMPLE
# Enter the matrix variable names (assumed to be Hermitian) (comma-separated):
# A0,A1,B0,B1
# Enter the monomial expressions (comma-separated; enter ^T for transpose, ^* for conjugate transpose, AB for A*B):
# A0,A1,B0,B1,I
# Enter a polynomial in terms of B0, A1, I, B1, A0, Adjoint:
# A0B0+A0B1+A1B0-A1B1
# Enter >= 0 constraints g(B0, A1, I, B1, A0) (comma-separated) (press Enter to finish):
# I-A0^2,I-A1^2,I-B0^2,I-B1^2,A0B0-B0A0,A0B1-B1A0,A1B0-B0A1,A1B1-B1A1
# Enter = I constraints g(B0, A1, I, B1, A0) (comma-separated) (press Enter to finish):
# A0^2, A1^2, B0^2, B1^2
# Lambda:
# 2.828427091787953