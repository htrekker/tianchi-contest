with open('test.tsv', 'r') as f:
    texts = f.readlines()
max_test = 0
for line in texts:
    line = line.split('\t')
    t_a = line[0].split()
    t_b = line[1].split()
    all = t_a + t_b
    for num in all:
        max_test = int(num) if int(num) > max_test else max_test

with open('train.tsv', 'r') as f:
    texts = f.readlines()
max_train = 0
for line in texts:
    line = line.split('\t')
    t_a = line[0].split()
    t_b = line[1].split()
    all = t_a + t_b
    for num in all:
        max_train = int(num) if int(num) > max_train else max_train
print(max_test, max_train)
