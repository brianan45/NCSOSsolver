import numpy as np
import sympy as sp

# clean_value takes an sp expression, np scalar, or np
# vector, and returns a rounded version of the input

def clean_value(val, zero_tol=1e-4, int_tol=1e-4):
    def is_near_zero(x): return abs(x) < zero_tol
    def is_near_int(x): return abs(x - round(x)) < int_tol

    def clean_number(x):
        if is_near_zero(x):
            return 0
        elif is_near_int(x):
            return float(round(x))
        return x
    # SymPy scalar
    if isinstance(val, sp.Basic):
        return val.replace(
            lambda e: e.is_Number,
            lambda e: clean_number(e)
        )

    # SymPy Matrix
    elif isinstance(val, sp.MatrixBase):
        return val.applyfunc(clean_number)

    # NumPy scalar
    elif isinstance(val, (int, float, np.number)):
        return clean_number(val)

    # NumPy array
    elif isinstance(val, np.ndarray):
        return np.vectorize(clean_number, otypes=[np.float64])(val)

    else:
        raise TypeError("Unsupported type. Provide a SymPy expression, NumPy scalar, or NumPy array.")