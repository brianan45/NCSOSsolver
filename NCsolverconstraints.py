import sympy as sp
from sympy.logic.boolalg import BooleanFalse
from sympy import Mul
import cvxpy as cp
import numpy as np
import sys
from NC_monomial_input import get_vars_vec
from NC_poly_input import get_poly
from sp_to_cp_constraint_multiQ import sp_to_cp_constraint_multiQ
from NCget_coeffs import get_coeffs
from NCget_constraints import get_constraints
from matrix_sqrt import matrix_sqrt
from clean_value import clean_value
from sub_identity_matrix import sub_identity_matrix, get_matrix_exps

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

n, matrix_vars, v = get_vars_vec()
# print("v =",v)
# print("type(v) =", type(v))
# print("vars =", vars)
# print("type(vars) =", type(matrix_vars))
# print("type(vars_elements) =", type(next(iter(matrix_vars))))
m = v.shape[0]
lm_sp = sp.Symbol("lm_sp")
# p = get_poly(matrix_vars, n)
p = get_poly(matrix_vars, n)
p += lm_sp * sp.Identity(n)
# p += lm_sp * sp.Identity(n)
# print("p =", p)
# print("type(p) =", type(p))
g_list, I_list = get_constraints(matrix_vars, n)

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q0 = cp.Variable((m, m), PSD=True)  # for main SOS
Qi_list = [cp.Variable((m, m), PSD=True) for _ in g_list]  # one per constraint
# print(Qi_list)
lm_cp = cp.Variable(name="lm_cp")

# Build the expression v^T Q v symbolically
Q0_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'q0_{min(i,j)}_{max(i,j)}'))
Qi_syms = [
    sp.Matrix(m, m, lambda i, j, k=k: sp.Symbol(f'q{k+1}_{min(i,j)}_{max(i,j)}'))
    for k in range(len(g_list))
]
# print(Q0_sym)
# for m in Qi_syms:
#     print(m)

# v^T * Q * v symbolic expansion
# v_adj = v.applyfunc(sp.Adjoint)
# print("v_adj =", v_adj)

def reverse_monovec(expr_matrix):
    result = expr_matrix.copy()
    for i in range(expr_matrix.rows):
        entry = expr_matrix[i, 0]
        if isinstance(entry, Mul):
            reversed_entry = Mul(*reversed(entry.args), evaluate=False)
            result[i, 0] = reversed_entry
    return result

poly_expr = ((reverse_monovec(v)).T * Q0_sym * v)[0, 0]
for g_sym, Qi_sym in zip(g_list, Qi_syms):
    # print("\n((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym =", ((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym)
    poly_expr += ((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym  # * ((reverse_monovec(v)).T * Qi_sym * v)[0, 0]
    # poly_expr += (v_adj.T * Qi_sym * v)[0, 0] * g_sym

# print("poly_expr before expansion =", poly_expr)
# poly_expr = sum(Q_sym[i, j] * sp.Adjoint(v[i]) * v[j] for i in range(m) for j in range(m))
poly_expr = sp.expand(poly_expr)  
# print("poly_expr after expansion =", poly_expr)
# # Convert both to polynomials
# poly_expr_poly = sp.Poly(poly_expr, *vars)
# target_poly = sp.Poly(p, *vars)

# print("type(vars) =", type(matrix_vars))
# print("poly_expr - p =", poly_expr - p)

matrix_exps = get_matrix_exps(I_list, matrix_vars)
simplified = sub_identity_matrix(sp.expand(poly_expr - p), matrix_exps)
print("poly_expr - p =", sp.expand(poly_expr - p))
print("simplified =", simplified)

# sol = get_coeffs(sp.expand(poly_expr - p), matrix_vars)
sol = get_coeffs(simplified, matrix_vars)
# print("sol =", sol)

# Check for False anywhere in the solution list
if any(isinstance(s, BooleanFalse) for s in sol):
    print("Problem is infeasible. Exiting.")
    sys.exit()

# print("Solution:")
# print(sol)

extra_subs = {lm_sp: lm_cp}

cvx_eqs = []
# for sym, expr in sol.items():
#     eq = sp.Eq(sym, expr)
#     cvx_eqs.append(sp_to_cp_constraint(eq, Q, extra_subs=extra_subs))

for eq in sol:
    # print("eq =", eq)
    cvx_eqs.append(sp_to_cp_constraint_multiQ(eq, [Q0] + Qi_list, extra_subs=extra_subs))

cvx_eqs.append(lm_cp >= 0)

# Print all converted CVXPY equations
# print("\nConverted CVXPY equations:")
# for e in cvx_eqs:
#     print(e)

problem = cp.Problem(cp.Minimize(lm_cp),cvx_eqs)
problem.solve(solver=cp.SCS, verbose=True)

if problem.status == "infeasible":
    print("Problem is infeasible. Exiting.")
    sys.exit()

# print("Q matrix:")
# print(clean_value(Q.value))
# print("type(Q.value) =", type(Q.value))

print("\nLambda:")
print(clean_value(lm_cp.value))

print("Q0 matrix =", clean_value(Q0.value))
Q0_sqrt = sp.Matrix(matrix_sqrt(clean_value(Q0.value)))
vQ0v_sqrt = Q0_sqrt.T @ v
# print("vQ0v_sqrt:", (vQ0v_sqrt))
vQ0v = (clean_value(sp.Adjoint(vQ0v_sqrt) * vQ0v_sqrt)[0,0])

SOS_decomp = vQ0v

# Add g_i * (v^T Qi v) terms
for i, g in enumerate(g_list):
    print("Qi matrix:", clean_value(Qi_list[i].value))
    Qi_sqrt = sp.Matrix(matrix_sqrt(Qi_list[i].value))
    vQv_sqrt = Qi_sqrt.T @ v
    vQv = (clean_value(sp.Adjoint(vQv_sqrt) * vQv_sqrt)[0,0])
    # print("vQv =", vQv)
    # print("g =", g)
    # print("sp.MatMul(vQv, g, evaluate=False) =", sp.MatMul(vQv, g, evaluate=False))
    SOS_decomp = sp.Add(SOS_decomp, sp.MatMul(vQv, g, evaluate=False), evaluate=False)

print("\nSOS decomposition:", SOS_decomp)

# CHSH PROBLEM
# p = -A0B0-A0B1-A1B0+A1B1
# g_i = I-A0^2,I-A1^2,I-B0^2,I-B1^2,A0B0-B0A0,A0B1-B1A0,A1B0-B0A1,A1B1-B1A1

# lm - X^2 S.T. X^2-1=0, LM = 1
# LM - X^2 - Y^2 S.T. X^2-1=0,Y^2-1=0, SHOULD GET LM = 2
# P = (1+X)^2 + (1-Y)^2 S.T. X^2-1=0,Y^2-1=0

# ASSERT N >= 2 FOR THE SIZE OF INPUT MATRICES

# TRY TRIVIAL EXAMPLES OF THE FORM P = (1-X)^2 + (1-Y)^2 + ... (A HUGE SOS)
# (FIRST WITH NO CONSTRAINTS, THEN ADD CONSTRAINTS)

# TRY THE SUBSTITUTION OPTIMIZATION, EX: IF X^2=1, REPLACE INSTANCES OF X^2 WITH 1
# CHECK MONIQUE LAURENT FOR HOW TO HANDLE EACH CONSTRAINT
# TRY >= AND =< (to represent equality) CONSTRAINTS AGAIN
# TRY INCREASING THE TOLERANCE
# STUDY THE PRIMAL AND DUAL
# VERIFY THE CONSTRAINTS BEING PASSED TO CVXPY
# VERIFY THE CONSTRAINTS BEING PASSED TO THE SOLVER

# TEST THE COMPARISON OF COEFFICIENTS

# test p = X
# test constraint X^2-1 vs 1-X^2
# answers should be same, systems of equations should flip signs

# correct form is 1 - A^2

# will A^3, A^4, etc. ever be replaced with I?