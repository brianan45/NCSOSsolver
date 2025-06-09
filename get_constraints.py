import sympy as sp
import re

def preprocess_input(expr_str):
    # Replace ^ with **
    expr_str = expr_str.replace('^', '**')

    # Replace standalone 'i' with 'I' (imaginary unit)
    expr_str = re.sub(r'(?<![\w])i(?![\w])', 'I', expr_str)

    # Insert * between variables/numbers and variables/parentheses (e.g. x2y → x*2*y)
    expr_str = re.sub(r'(?<=[0-9a-zA-Z)])(?=[a-zA-Z(])', '*', expr_str)

    # Insert * between I and following variable/number/parenthesis (e.g. 3Ix → 3*I*x)
    expr_str = re.sub(r'(?<=[^*])I(?=[a-zA-Z0-9(])', 'I*', expr_str)

    return expr_str

def get_constraints(vars):
    var_dict = {str(v): v for v in vars}
    var_dict['I'] = sp.I  # Make sure 'I' is recognized as imaginary unit
    constraints = []

    while True:
        inp = input(f"Enter >= 0 constraints g({', '.join(str(v) for v in vars)}) (comma-separated) (press Enter to finish): ").strip()
        if inp == "":
            break

        exprs = inp.split(',')

        for expr_str in exprs:
            expr_str = preprocess_input(expr_str.strip())

            try:
                g = sp.sympify(expr_str, locals=var_dict)
                constraints.append(g)
            except Exception as e:
                print(f"Could not parse constraint '{expr_str}': {e}")
    return constraints