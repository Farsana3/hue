# Example program using simulated do-while, for loop, break, and continue

while True:  # Simulated do-while loop
    print("\nMenu:")
    print("1. Print even numbers from 1 to 10")
    print("2. Print numbers from 1 to 5, skipping 3")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        print("Even numbers from 1 to 10:")
        for i in range(1, 11):
            if i % 2 != 0:
                continue  # Skip odd numbers
            print(i, end=" ")
        print()

    elif choice == '2':
        print("Numbers from 1 to 5 (skipping 3):")
        for i in range(1, 6):
            if i == 3:
                continue  # Skip number 3
            print(i, end=" ")
        print()

    elif choice == '3':
        print("Exiting program.")
        break  # Exit the simulated do-while loop

    else:
        print("Invalid choice. Please try again.")
