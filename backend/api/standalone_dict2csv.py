import csv  # Import the csv module

the_dict = {'a': 1, 'b': 2, 'c': 3}
with open('dict.csv', 'w', newline='') as csv_file:  # Added newline='' for better compatibility across different systems
    writer = csv.writer(csv_file)
    for key, value in the_dict.items():
        writer.writerow([key, value])
