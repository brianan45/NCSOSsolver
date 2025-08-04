import sympy as sp
from sympy.logic.boolalg import BooleanFalse
import cvxpy as cp
import numpy as np
import sys
from NCmonomial_input import get_vars_vec
from NCpoly_input import get_poly
from sp_to_cp_constraint import sp_to_cp_constraint
from NCget_coeffs import get_coeffs
from matrix_sqrt import matrix_sqrt
from clean_value import clean_value

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

n, matrix_vars, v = get_vars_vec()
m = v.shape[0]
lm_sp = sp.Symbol("lm_sp")
p = get_poly(matrix_vars, n)
p += lm_sp * sp.Identity(n)

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q = cp.Variable((m, m), PSD=True)
lm_cp = cp.Variable(name="lm_cp")

# Build the expression v^T Q v symbolically
Q_sym = sp.Matrix(Q.shape[0], Q.shape[1], lambda i, j: sp.Symbol(f'q{min(i,j)}{max(i,j)}'))

# v^T * Q * v symbolic expansion
v_adj = v.applyfunc(sp.Adjoint)
poly_expr = (v_adj.T * Q_sym * v)[0, 0]
poly_expr = sp.expand(poly_expr)  

sol = get_coeffs(sp.expand(poly_expr - p), matrix_vars)

# Check for False anywhere in the solution list
if any(isinstance(s, BooleanFalse) for s in sol):
    print("Problem is infeasible. Exiting.")
    sys.exit()

extra_subs = {lm_sp: lm_cp}

cvx_eqs = []

for eq in sol:
    cvx_eqs.append(sp_to_cp_constraint(eq, Q, extra_subs=extra_subs))

cvx_eqs.append(lm_cp >= 0)

problem = cp.Problem(cp.Minimize(lm_cp),cvx_eqs)
problem.solve(solver=cp.CVXOPT)

if problem.status == "infeasible":
    print("Problem is infeasible. Exiting.")
    sys.exit()

print("Q matrix:")
print(clean_value(Q.value))

print("\nLambda:")
print(clean_value(lm_cp.value))

Q_sqrt = sp.Matrix(matrix_sqrt(Q.value))
vQv_sqrt = Q_sqrt.T @ v
sos_expr = clean_value((vQv_sqrt.T * vQv_sqrt)[0,0])
print("SOS decomposition:")
print(sos_expr)

