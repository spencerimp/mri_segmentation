import csv
import numpy as np

ignored_labels = range(1,4)+range(5,11)+range(12,23)+range(24,30)+[33,34]+[42,43]+[53,54]+range(63,69)+[70,74]+\
                    range(80,100)+[110,111]+[126,127]+[130,131]+[158,159]+[188,189]

true_labels = [4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57,
                58, 59, 60, 61, 62, 69, 71, 72, 73, 75, 76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129, 132, 133, 134, 135, 136,
                137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
                179, 180, 181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                201, 202, 203, 204, 205, 206, 207]

OASIS_label_path = './MICCAI-Challenge-2012-Label-Information_v2.csv'
MICCAI_label_path = './MICCAI-Challenge-2012-Label-Information_v3.csv'
with open(OASIS_label_path) as f:
    csvReader = csv.reader(f, delimiter=',')
    labels, names = zip(*[row for row in csvReader])

labels = np.array(map(int, labels))
names = np.array(names)

for ignored_label in ignored_labels:
    labels[np.where(labels==ignored_label)] = 0

idx = 1
miccai_names = []
for true_label in true_labels:
    miccai_name = names[np.where(labels==true_label)][0]
    labels[np.where(labels==true_label)] = idx

    miccai_names.append(miccai_name)
    idx += 1

#idx = 135
miccai_names = np.array(miccai_names)[:idx-1]
miccai_labels = range(1, 1+len(true_labels))

# output to file
with open(MICCAI_label_path, 'wb') as f:
    csvWriter = csv.writer(f, delimiter=',')
    csvWriter.writerows(zip(miccai_labels, miccai_names))

