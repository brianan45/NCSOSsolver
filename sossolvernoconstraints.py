import sympy as sp
import cvxpy as cp
import numpy as np
import sys
from monomial_input import get_monomial_vector
from poly_input import get_polynomial_from_input
from sp_to_cp_constraint import sp_to_cp_constraint
from get_coeffs import get_coeffs
from matrix_sqrt import matrix_sqrt
from clean_value import clean_value

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

v, vars = get_monomial_vector()
# print(v)
# print(vars)
lm_sp = sp.Symbol("lm_sp")
p = get_polynomial_from_input(vars) + lm_sp
# print(p)
m = v.rows

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q = cp.Variable((m, m), PSD=True)
lm_cp = cp.Variable(name="lm_cp")

# Build the expression v^T Q v symbolically
Q_sym = sp.Matrix(Q.shape[0], Q.shape[1], lambda i, j: sp.Symbol(f'q{min(i,j)}{max(i,j)}'))

# v^T * Q * v symbolic expansion
poly_expr = ((v.T * Q_sym * v)[0, 0]).expand()
print(poly_expr)

# Convert both to polynomials
poly_expr_poly = sp.Poly(poly_expr, *vars)
target_poly = sp.Poly(p, *vars)

sol = get_coeffs(poly_expr_poly,target_poly,vars)
# print("Solution:")
# print(sol)

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
problem.solve(solver=cp.CVXOPT)

if problem.status == "infeasible":
    print("Problem is infeasible. Exiting.")
    sys.exit()

print("Q matrix:")
print(clean_value(Q.value))

print("\nLambda:")
print(clean_value(lm_cp.value))

Q_sqrt = matrix_sqrt(Q.value)

sos_expr = (v.T @ Q_sqrt @ Q_sqrt.T @ v)[0, 0]
print("\n SOS decomposition:")
print(clean_value(sp.expand(sos_expr)))