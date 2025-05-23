import sympy as sp
import cvxpy as cp
import numpy as np

# How to find Q if there are multiple possible Qs?
# Should we provide an expression representing the possible Qs?
# What counts as an optimal Q?

# algorithm:
# given: p(x) and v
# formulate as SDP program: min lambda s.t. vT*Q*v = p(x) + lambda, Q SDP
# compute vT*Q*v with placeholders for entries of Q
# solve by setting coefficients of p(x) + lambda equal to vT*Q*v, thereby finding entries of Q
# if lambda = 0, original p(x) is an SOS
# find Cholesky decomp of Q
# p(x) = (LT^v)^2