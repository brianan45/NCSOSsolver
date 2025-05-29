import sympy as sp
import re

def get_constraints(vars):
    var_dict = {str(v): v for v in vars}
    constraints = []

    while True:
        inp = input(f"Enter equality-with-0 constraints g({', '.join(str(v) for v in vars)}) (comma-separated) (press Enter to finish): ").strip()
        if inp == "":
            break
        inp = inp.replace("^", "**")
        inp = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', inp)

        try:
            g = sp.sympify(inp, locals=var_dict)
            constraints.append(g)
        except Exception as e:
            print(f"Could not parse constraint: {e}")
    return constraints
