import sympy as sp
import cvxpy as cp
import numpy as np
import sys
from monomial_input import get_monomial_vector
from poly_input import get_polynomial_from_input
from sp_to_cp_constraint import sp_to_cp_constraint
from get_coeffs import get_coeffs
from matrix_sqrt import matrix_sqrt
from get_constraints import get_constraints

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

v, vars = get_monomial_vector()
# print(v)
# print(vars)
lm_sp = sp.Symbol("lm_sp")
p = get_polynomial_from_input(vars) + lm_sp
g_list = get_constraints(vars)
# print(p)
m = v.rows

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q0 = cp.Variable((m, m), PSD=True)  # for main SOS
Qi_list = [cp.Variable((m, m), PSD=True) for _ in g_list]  # one per constraint
lm_cp = cp.Variable(name="lm_cp")

# Symbolic Q matrices for matching coefficients
Q0_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'q0_{min(i,j)}{max(i,j)}'))
Qi_syms = [
    sp.Matrix(m, m, lambda i, j: sp.Symbol(f'q{i+1}_{min(i,j)}{max(i,j)}'))
    for i in range(len(g_list))
]

# v^T * Q * v symbolic expansion
# Build the symbolic polynomial: v^T Q0 v + sum g_i * (v^T Qi v)
poly_expr = (v.T * Q0_sym * v)[0, 0]
for g_sym, Qi_sym in zip(g_list, Qi_syms):
    sos_poly += g_sym * (v.T * Qi_sym * v)[0, 0]
print(poly_expr)

# Convert both to polynomials
poly_expr_poly = sp.Poly(poly_expr, *vars)
target_poly = sp.Poly(p, *vars)

sol = get_coeffs(poly_expr_poly,target_poly,vars)
print("Solution:")
print(sol)

if not sol:
    print("No solution; try a different monomial vector")
    sys.exit()

extra_subs = {lm_sp: lm_cp}

cvx_eqs = []
for sym, expr in sol.items():
    eq = sp.Eq(sym, expr)
    cvx_eqs.append(sp_to_cp_constraint(eq, Q, extra_subs=extra_subs))

cvx_eqs.append(lm_cp >= 0)

# Print all converted CVXPY equations
for e in cvx_eqs:
    print(e)

problem = cp.Problem(cp.Minimize(lm_cp),cvx_eqs)
problem.solve()

if problem.status == "infeasible":
    print("Problem is infeasible. Exiting.")
    sys.exit()

# print("Q matrix:")
# print(Q.value)

print("\nLambda:")
print(lm_cp.value)

Q_sqrt = matrix_sqrt(Q.value)

print("\nSOS decomposition:")
print(np.dot(v.T @ Q_sqrt,Q_sqrt.T @ v))