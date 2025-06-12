import sympy as sp
import cvxpy as cp
import numpy as np
import sys
from monomial_input import get_monomial_vector
from complex_poly_input import get_polynomial_from_input
from sp_to_cp_constraint_multiQ import sp_to_cp_constraint_multiQ
from get_coeffs import get_coeffs
from matrix_sqrt import matrix_sqrt
from clean_value import clean_value

# An SOS solver that takes a set of variables, a vector v of monomials
# in those variables, a complex polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

v, vars = get_monomial_vector()
# print(v)
# print(vars)
lm_sp = sp.Symbol("lm_sp", real=True)
p = get_polynomial_from_input(vars) + lm_sp
print("p(x)=", p)
p_r = sp.re(p)
p_i = sp.im(p)
m = v.rows
i = sp.I

# Define symmetric 4x4 matrix Q and lambda with cvxpy variables
Q0 = cp.Variable((m, m), PSD=True)  # matrix representing real
Q1 = cp.Variable((m, m), PSD=True)  # imaginary

Q2 = cp.vstack([cp.hstack([Q0,Q1]), cp.hstack([-Q1,Q0])]) 

lm_cp = cp.Variable(name="lm_cp")

# Symbolic Q matrices for matching coefficients
Q0_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'Q0_{min(i,j)}{max(i,j)}'))
Q1_sym = sp.Matrix(m, m, lambda i, j: sp.Symbol(f'Q1_{min(i,j)}{max(i,j)}'))

# print(Q0_sym)
# for m in Q1_syms:
#     print(m)

# v^T * Q * v symbolic expansion
# Build the symbolic polynomial: v^T Q0 v + sum g_i * (v^T Q1 v)

poly_r = (v.T * Q0_sym * v)[0, 0]
print(poly_r)

poly_i = (v.T * Q1_sym * v)[0, 0]
print(poly_i)

# Convert both to polynomials
sp_poly_r = sp.Poly(poly_r, *vars)
target_poly_r = sp.Poly(p_r, *vars)

sp_poly_i = sp.Poly(poly_i, *vars)
target_poly_i = sp.Poly(p_i, *vars)
print(sp_poly_r, "\n", target_poly_r, "\n", sp_poly_i, "\n", target_poly_i)

sol_r = get_coeffs(sp_poly_r,target_poly_r,vars)
# print(sol_r)
sol_i = get_coeffs(sp_poly_i,target_poly_i,vars)
# print(sol_i)
# print(type(sol_r),type(sol_i))
sol = {**(sol_r or {}), **(sol_i or {})}
print("Solution:")
print(sol)

if not sol:
    print("No solution; try a different monomial vector")
    sys.exit()

# extra_subs = {lm_sp: lm_cp}

cvx_eqs = []
for sym, expr in sol.items():
    eq = sp.Eq(sym, expr)
    cvx_eqs.append(sp_to_cp_constraint_multiQ(eq, [Q0] + [Q1], extra_subs={lm_sp: lm_cp}))

cvx_eqs.append(Q2 >> 0)
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
print(clean_value(Q1.value))
print(clean_value(Q2.value))

SOS_decomp = (v.T * clean_value(Q0.value) * v)[0, 0] + i*(v.T * clean_value(Q1.value) * v)[0, 0] # v^T Q0 v

print("\nSOS decomposition of p(x) + λ:")
print(SOS_decomp)