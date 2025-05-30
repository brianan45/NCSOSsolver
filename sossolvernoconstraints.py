import sympy as sp
import cvxpy as cp
import numpy as np
import sys
from monomial_input import get_monomial_vector
from poly_input import get_polynomial_from_input
from sp_to_cp_constraint import sp_to_cp_constraint
from get_coeffs import get_coeffs
from matrix_sqrt import matrix_sqrt

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
problem.solve()

if problem.status == "infeasible":
    print("Problem is infeasible. Exiting.")
    sys.exit()

def simp_matrix(matrix, tol=1e-5):
    return np.where(np.abs(matrix) < tol, 0, matrix)

def clean_expr(expr, tol=1e-5):
    return expr.replace(
        lambda e: e.is_Number,
        lambda e: 0 if abs(e) < tol else (round(e) if abs(e - round(e)) < tol else e)
    )

def clean_np(obj, zero_tol=1e-10, int_tol=1e-5):
    def clean_val(x):
        if abs(x) < zero_tol:
            return 0
        elif abs(x - round(x)) < int_tol:
            return round(x)
        return x

    if isinstance(obj, (int, float, np.number)):
        return clean_val(obj)
    elif isinstance(obj, np.ndarray):
        return np.vectorize(clean_val)(obj)
    else:
        raise TypeError("Input must be a number or a NumPy array.")

print("Q matrix:")
print(simp_matrix(Q.value))

# print("\nLambda:")
# print(simplify_expr(lm_cp.value))

print("\nLambda:")
print(0 if abs(lm_cp.value) < 1e-5 else lm_cp.value)

Q_sqrt = matrix_sqrt(Q.value)

print(clean_np(lm_cp.value))
print(clean_np(Q.value))

sos_expr = (v.T @ Q_sqrt @ Q_sqrt.T @ v)[0, 0]
sos_simp = clean_expr(sp.expand(sos_expr))
print("\nSimplified SOS decomposition:")
print(sos_simp)