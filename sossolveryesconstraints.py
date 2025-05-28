import sympy as sp
import cvxpy as cp
import numpy as np
import sys
from monomial_input import get_monomial_vector
from poly_input import get_polynomial_from_input
from sp_to_cp_constraint import sp_to_cp_constraint
from get_coeffs import get_coeffs

v, vars = get_monomial_vector()
print(v)
print(vars)
p = get_polynomial_from_input(vars)
print(p)
m = v.rows

# Define symmetric 4x4 matrix Q with cvxpy variables
Q = cp.Variable((m, m), PSD=True)

# Build the expression v^T Q v symbolically
Q_sym = sp.Matrix(Q.shape[0], Q.shape[1], lambda i, j: sp.Symbol(f'q{min(i,j)}{max(i,j)}'))

# v^T * Q * v symbolic expansion
poly_expr = (v.T * Q_sym * v)[0, 0].expand()
print(poly_expr)
# print((Q_sym * v)[0, 0].expand())
# print((v.T * Q_sym)[0, 0].expand())

# Convert both to polynomials
poly_expr_poly = sp.Poly(poly_expr, *vars)
target_poly = sp.Poly(p, *vars)

sol = get_coeffs(poly_expr_poly,target_poly,vars)
print("Solution:")
print(sol)

if not sol:
    print("No solution; try a different monomial vector")
    sys.exit()

cvx_eqs = []
for sym, expr in sol.items():
    eq = sp.Eq(sym, expr)
    cvx_eqs.append(sp_to_cp_constraint(eq, Q))

# Print all converted CVXPY equations
for e in cvx_eqs:
    print(e)

problem = cp.Problem(cp.Minimize(1),cvx_eqs)
problem.solve()

print("Q matrix:")
print(Q.value)

L = np.linalg.cholesky(Q.value)
print("\n", np.round(L,4))

print("\nSOS decomposition:")
print(np.dot(v.T @ L,L.T @ v))