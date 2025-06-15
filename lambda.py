# List of tuples
students = [("Alice", 88), ("Bob", 72), ("Charlie", 95), ("David", 85)]

# Sort by the second item using lambda
sorted_students = sorted(students, key=lambda x: x[1])

# Print the sorted list
print("Sorted by score:", sorted_students)
