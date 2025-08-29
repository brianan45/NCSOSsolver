This repo contains solvers that rewrite polynomials as sum-of-squares using semidefinite optimization, plus required helper functions. CVXPY and SymPy are required to run the solvers; both can be downloaded by entering "pip install sympy cvxpy" in the command line. Included in the repo is a guide, Final_Report.pdf, on how to use the solvers and more context behind them.

Polynomial solvers include:
unconstrained commutative (sossolvernoconstraints.py)
constrained commutative (sossolveryesconstraints.py)
unconstrained complex commutative (complexsolver.py)
unconstrained non-commutative (NCsolvernoconstraints.py)
constrained non-commutative (NCsolverconstraints.py)

An example of how to use the constrained non-commutative solver (NCsovlerconstraints.py) can be seen below:

Enter the matrix variable names (assumed to be Hermitian) (comma-separated):
A0,A1,B0,B1
Enter the monomial expressions (comma-separated; enter ^T for transpose, ^* for conjugate transpose, AB for A*B):
A0,A1,B0,B1
Enter a polynomial in terms of B0, A1, I, B1, A0, Adjoint:
A0B0+A0B1+A1B0-A1B1
Enter = 0 constraints g(B0, A1, I, B1, A0) (comma-separated) (press Enter to finish):
I-A0^2,I-A1^2,I-B0^2,I-B1^2,A0B0-B0A0,A0B1-B1A0,A1B0-B0A1,A1B1-B1A1
Enter = I constraints g(B0, A1, I, B1, A0) (comma-separated) (press Enter to finish):
A0^2, A1^2, B0^2, B1^2
Lambda:
2.828427091787953

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC)(ALLRP-578455-2022) as well as the support of the Air Force Office of Scientific Research under award number FA9550-20-1-0375. The supervision of Prof. Anne Broadbent and Dr. Denis Rochette, both of the University of Ottawa, was immensely helpful throughout the completion of this project.
