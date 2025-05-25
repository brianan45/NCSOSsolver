import sympy as sp
import cvxpy as cp
import numpy as np
from soltoconstraints import soltoconstraints

# Define symbolic variable
x = sp.Symbol('x')

# Define monomial basis vector v = [1, x, x^2, x^3]
v = sp.Matrix([1, x, x**2, x**3])

# Target polynomial p(x)
p = x**6 + 3*x**4 + 3*x**2 + 1

# Define symmetric 4x4 matrix Q with cvxpy variables
Q = cp.Variable((4, 4), PSD=True)
print(Q[0,0])

# Build the expression v^T Q v symbolically
Q_sym = sp.Matrix(Q.shape[0], Q.shape[1], lambda i, j: sp.Symbol(f'q{min(i,j)}{max(i,j)}'))
# sp.Matrix necessary to compute v.T * Q_sym * v

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

# Fill in Q matrix with solved values
# Q_num = np.zeros((4, 4))
# for i in range(4):
#     for j in range(i, 4):
#         key = f'q{i}{j}'
#         print(key)
#         val = sol.get(sp.Symbol(key), sol.get(sp.Symbol(f'q{j}{i}'), 0))
#         print(val)
#         Q_num[i, j] = Q_num[j, i] = val

# make a trivial objective function
# one constraint will be Q PSD
# other constraints come from polynomial equality
# extract variables of Q



# Print Q
print("Q matrix:")
print(np.round(Q_sym, 4))

# Cholesky decomposition (to get SOS)
R = np.linalg.cholesky(Q_sym)

print("\nCholesky factor R:")
print(np.round(R, 4))

# Show SOS decomposition
print("\nSOS decomposition terms (rows of R · v):")
for i in range(4):
    row = R[i, :]
    poly = sum(row[j] * v[j] for j in range(4))
    print(f"({sp.simplify(poly)})^2")
