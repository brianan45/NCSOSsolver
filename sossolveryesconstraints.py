import sympy as sp
import cvxpy as cp
import numpy as np
import sys
from monomial_input import get_monomial_vector
from poly_input import get_polynomial_from_input
from sp_to_cp_constraint_multiQ import sp_to_cp_constraint_multiQ
from get_coeffs import get_coeffs
from matrix_sqrt import matrix_sqrt
from get_constraints import get_constraints
from clean_value import clean_value
from is_feasible import is_feasible

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

# Issues: solving x^n for large, solving motzkin poly, solving x with -x >= 0, solving x+1 with -x+1 >= 0
# solving for x with x-1 >= 0, -x-1 >= 0 (no feasible region) yields lambda = 0 rather than infeasible
# solve x^2-100 s.t. x^2-50 >= 0 -> x^2-50; fix output to get Putinar form

v, vars = get_monomial_vector()
# print(v)
# print(vars)
lm_sp = sp.Symbol("lm_sp")
p = get_polynomial_from_input(vars) + lm_sp
print("p(x)=",p)
g_list = get_constraints(vars)
print("Constraints: ", g_list)
m = v.rows

feasible = is_feasible(g_list,vars)
print(feasible)

if not feasible:
    print("Problem has no feasible region")
    sys.exit()

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q0 = cp.Variable((m, m), PSD=True)  # for main SOS
Qi_list = [cp.Variable((m, m), PSD=True) for _ in g_list]  # one per constraint
print(Qi_list)
lm_cp = cp.Variable(name="lm_cp")

# Symbolic Q matrices for matching coefficients
Q0_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'q0_{min(i,j)}{max(i,j)}'))
Qi_syms = [
    sp.Matrix(m, m, lambda i, j, k=k: sp.Symbol(f'q{k+1}_{min(i,j)}{max(i,j)}'))
    for k in range(len(g_list))
]
print(Q0_sym)
for m in Qi_syms:
    print(m)

# v^T * Q * v symbolic expansion
# Build the symbolic polynomial: v^T Q0 v + sum g_i * (v^T Qi v)
poly_expr = (v.T * Q0_sym * v)[0, 0]
for g_sym, Qi_sym in zip(g_list, Qi_syms):
    poly_expr += g_sym * (v.T * Qi_sym * v)[0, 0]
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
    cvx_eqs.append(sp_to_cp_constraint_multiQ(eq, [Q0] + Qi_list, extra_subs=extra_subs))

cvx_eqs.append(lm_cp >= 0)

# Print all converted CVXPY equations
for e in cvx_eqs:
    print(e)

problem = cp.Problem(cp.Minimize(lm_cp),cvx_eqs)
problem.solve(solver=cp.SCS)
print(problem.status)

if problem.status != "optimal":
    print("Problem is infeasible. Exiting.")
    sys.exit()

# print("Q matrix:")
# print(Q.value)

print("\nLambda:")
print(clean_value(lm_cp.value))
print("\nQ Matrices:")
print(clean_value(Q0.value))
for Q in Qi_list:
    print(clean_value(Q.value))

SOS_decomp = (v.T * clean_value(Q0.value) * v)[0, 0]  # v^T Q0 v

# Add g_i * (v^T Qi v) terms
for i, g in enumerate(g_list):
    Qi_val = clean_value(Qi_list[i].value)
    SOS_decomp = sp.Add(SOS_decomp, sp.Mul(g, (v.T * Qi_val * v)[0, 0], evaluate=False), evaluate=False)

print("\nSOS decomposition of p(x) + λ:")
print(SOS_decomp)