import csv


title = "This is a test stentence"

with open('test.txt','a+') as f:
    f.write(title)
    f.close()


with open('2.csv','r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        print(row)
        print(row[0])
output_list = ['1','2','3','4']
with open('2.csv','a+',encoding='utf-8',newline='') as csvfile:
    w = csv.writer(csvfile)
    w.writerow(output_list)