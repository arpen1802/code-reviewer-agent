# sample_code.py
# A Python file to test the reviewer agent.

def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)


def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(len(lst)):
            if lst[i] == lst[j]:
                duplicates.append(lst[i])
    return duplicates


def read_config(filename):
    f = open(filename)
    data = f.read()
    return data


result = calculate_average([10, 20, 30])
print("Average:", result)

dupes = find_duplicates([1, 2, 3, 2, 4, 1])
print("Duplicates:", dupes)
