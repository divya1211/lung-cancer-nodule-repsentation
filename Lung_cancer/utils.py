import csv

def find(nodule_id, patient_id):
    my_list = []
    csv_file = csv.reader(open('characteristics.csv', "r"), delimiter=",")
    for row in csv_file:
         if patient_id == row[0] and nodule_id == row[2]:
                return int(row[11])
