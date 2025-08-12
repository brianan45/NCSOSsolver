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
from remove_zero_terms import remove_zero_terms

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

v, vars = get_monomial_vector()
lm_sp = sp.Symbol("lm_sp")
p = get_polynomial_from_input(vars) + lm_sp
g_list = get_constraints(vars)
m = v.rows

feasible = is_feasible(g_list,vars)

if not feasible:
    print("Problem has no feasible region")
    sys.exit()

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q0 = cp.Variable((m, m), PSD=True)  # for main SOS
Qi_list = [cp.Variable((m, m), PSD=True) for _ in g_list]  # one per constraint
lm_cp = cp.Variable(name="lm_cp")

# Symbolic Q matrices for matching coefficients
Q0_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'q0_{min(i,j)}_{max(i,j)}'))
Qi_syms = [
    sp.Matrix(m, m, lambda i, j, k=k: sp.Symbol(f'q{k+1}_{min(i,j)}_{max(i,j)}'))
    for k in range(len(g_list))
]

# v^T * Q * v symbolic expansion
# Build the symbolic polynomial: v^T Q0 v + sum g_i * (v^T Qi v)
poly_expr = (v.T * Q0_sym * v)[0, 0]
for g_sym, Qi_sym in zip(g_list, Qi_syms):
    poly_expr += g_sym * (v.T * Qi_sym * v)[0, 0]

# Convert both to polynomials
poly_expr_poly = sp.Poly(poly_expr, *vars)
target_poly = sp.Poly(p, *vars)

# Compare coefficients term by term to get constraints for entries of Q_i
sol = get_coeffs(poly_expr_poly,target_poly,vars)

if not sol:
    print("No solution; try a different monomial vector")
    sys.exit()

extra_subs = {lm_sp: lm_cp}

cvx_eqs = []

# Convert the solution to cvxpy constraints
for sym, expr in sol.items():
    eq = sp.Eq(sym, expr)
    cvx_eqs.append(sp_to_cp_constraint_multiQ(eq, [Q0] + Qi_list, extra_subs=extra_subs))

cvx_eqs.append(lm_cp >= 0)

problem = cp.Problem(cp.Minimize(lm_cp),cvx_eqs)
problem.solve(solver=cp.SCS)

if problem.status != "optimal":
    print("Problem is infeasible. Exiting.")
    sys.exit()

print("\nLambda:")
print(clean_value(lm_cp.value))
print("\nQ Matrices:")
print(clean_value(Q0.value))
for Q in Qi_list:
    print(clean_value(Q.value))

Q0_sqrt = matrix_sqrt(clean_value(Q0.value))
vQ0v_sqrt = Q0_sqrt.T @ v
vQ0v = clean_value((vQ0v_sqrt.T * vQ0v_sqrt)[0,0])

SOS_decomp = vQ0v

# Add g_i * (v^T Qi v) terms
for i, g in enumerate(g_list):
    Qi_sqrt = clean_value(matrix_sqrt(Qi_list[i].value))
    vQv_sqrt = Qi_sqrt.T @ v
    vQv = remove_zero_terms(clean_value((vQv_sqrt.T * vQv_sqrt)[0,0]))
    SOS_decomp = sp.Add(SOS_decomp, sp.Mul(g, vQv, evaluate=False), evaluate=False)

print("\nSOS decomposition of p(x) + λ:")
print(SOS_decomp)