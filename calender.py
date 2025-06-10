import calendar
month = int(input("Enter month : "))
year = int(input("Enter year : "))

# Print the calendar
print("\nCalendar with calendar module:")
print(calendar.month(year, month))



import datetime


month = int(input("Enter month : "))
year = int(input("Enter year : "))

print("Without calendar module:")
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("\n", "  ".join(days))


first_day = datetime.date(year, month, 1)
start_day = first_day.weekday()  # Monday = 0, Sunday = 6


if month == 12:
    next_month = datetime.date(year + 1, 1, 1)
else:
    next_month = datetime.date(year, month + 1, 1)

last_day = (next_month - datetime.timedelta(days=1)).day

print("    " * (start_day+1), end="")  # padding for first week

for day in range(1, last_day + 1):
    print(f"{ day: >3}", end="  ")
    current_day = (start_day + day - 1) % 7
    if current_day == 6:
        print()  # New line after Sunday