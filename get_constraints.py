import sympy as sp
import re

def get_constraints(vars):
    var_dict = {str(v): v for v in vars}
    constraints = []

    while True:
        inp = input(f"Enter greater-than-0 constraints g({', '.join(str(v) for v in vars)}) (comma-separated) (press Enter to finish): ").strip()
        if inp == "":
            break

        # Split by comma and preprocess each expression
        exprs = inp.split(',')

        for expr_str in exprs:
            expr_str = expr_str.strip()
            expr_str = expr_str.replace("^", "**")
            expr_str = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', expr_str)

            try:
                g = sp.sympify(expr_str, locals=var_dict)
                constraints.append(g)
            except Exception as e:
                print(f"Could not parse constraint '{expr_str}': {e}")
    return constraints
