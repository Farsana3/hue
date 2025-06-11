def bitwise_operations(a: int, b: int):
    print(f"Input values: a = {a} ({bin(a)}), b = {b} ({bin(b)})")

    left_shift_a = a << 1
    print(f"Left Shift a << 1: {left_shift_a} ({bin(left_shift_a)})")

    right_shift_b = b >> 1
    print(f"Right Shift b >> 1: {right_shift_b} ({bin(right_shift_b)})")

    xor_result = a ^ b
    print(f"XOR a ^ b: {xor_result} ({bin(xor_result)})")

if __name__ == "__main__":
    a = 12
    b = 5
    bitwise_operations(a, b)
