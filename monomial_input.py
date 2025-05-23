import sympy as sp

def get_monomial_vector():
    # Step 1: Get variables from the user
    var_input = input("Enter the variables (comma-separated): ")
    variables = sp.symbols(var_input)  # Convert the input string to sympy symbols

    # Step 2: Get the monomials from the user
    monomial_input = input(f"Enter the monomials in terms of {', '.join([str(v) for v in variables])} (comma-separated): ")

    # Step 3: Parse the monomials into sympy expressions
    try:
        monomial_vector = [sp.sympify(m.strip(), locals={str(variables[i]): variables[i] for i in range(len(variables))}) 
                           for m in monomial_input.split(',')]
        print("Parsed monomial vector:", monomial_vector)
        return monomial_vector, variables
    except Exception as e:
        print("Error parsing input:", e)
        return [], []

# Example of using the function
monomial_vector, variables = get_monomial_vector()

# Optionally print the monomial vector and variables
print("Monomial vector:", monomial_vector)
print("Variables:", variables)