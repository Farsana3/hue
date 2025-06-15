def main():
    # List: a list of student names
    students = ["Alice", "Bob", "Charlie", "David"]

    # Tuple: immutable subject codes
    subjects = ("Math", "Science", "History")

    # Set: a set of unique student IDs (duplicates are removed)
    student_ids = {101, 102, 103, 101}

    # Dictionary: student name to grade mapping
    grades = {
        "Alice": 88,
        "Bob": 75,
        "Charlie": 93
    }

    # Display all
    print("List of students:", students)
    print("Tuple of subjects:", subjects)
    print("Set of unique student IDs:", student_ids)
    print("Dictionary of grades:", grades)

    # Accessing elements
    print("\nAccessing elements:")
    print("First student in list:", students[0])
    print("First subject in tuple:", subjects[0])
    print("Charlie's grade:", grades.get("Charlie"))

    # Demonstrate adding to list, set, and dictionary
    students.append("Eve")
    student_ids.add(104)
    grades["David"] = 85

    print("\nAfter adding new entries:")
    print("Updated students list:", students)
    print("Updated student IDs set:", student_ids)
    print("Updated grades dictionary:", grades)

if __name__ == "__main__":
    main()
