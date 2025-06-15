# Define functions for each arithmetic operation

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b != 0:
        return a / b
    else:
        return "Error: Division by zero"

def modulus(a, b):
    if b != 0:
        return a % b
    else:
        return "Error: Modulus by zero"

# Main program
def main():
    try:
        # Taking user input
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))

        # Performing operations
        print(f"\nResults:")
        print(f"Addition: {add(num1, num2)}")
        print(f"Subtraction: {subtract(num1, num2)}")
        print(f"Multiplication: {multiply(num1, num2)}")
        print(f"Division: {divide(num1, num2)}")
        print(f"Modulus: {modulus(num1, num2)}")

    except ValueError:
        print("Invalid input! Please enter numeric values.")

# Run the program
if __name__ == "__main__":
    main()
