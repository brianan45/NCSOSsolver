import sympy as sp
import cvxpy as cp
import numpy as np
from sp_to_cp_constraint import sp_to_cp_constraint

# Define symbolic variable
x = sp.Symbol('x')

# Define monomial basis vector v = [1, x, x^2, x^3]
v = sp.Matrix([1, x, x**2, x**3])

# Target polynomial p(x)
p = x**6 + 3*x**4 + 3*x**2 + 1

# Define symmetric 4x4 matrix Q with cvxpy variables
Q = cp.Variable((4, 4), PSD=True)

# Build the expression v^T Q v symbolically
Q_sym = sp.Matrix(Q.shape[0], Q.shape[1], lambda i, j: sp.Symbol(f'q{min(i,j)}{max(i,j)}'))

# v^T * Q * v symbolic expansion
poly_expr = (v.T * Q_sym * v)[0, 0].expand()
print(poly_expr)

# Extract coefficients from polynomial expression
coeffs_expr = sp.Poly(poly_expr, x).all_coeffs()
coeffs_target = sp.Poly(p, x).all_coeffs()
print(coeffs_expr)
print(coeffs_target)

# Match coefficients to get linear equations
eqns = [sp.Eq(e1, e2) for e1, e2 in zip(coeffs_expr, coeffs_target)]
print(eqns)

# Solve symbolically to get values for q_ij
sol = sp.solve(eqns)
print(sol)


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