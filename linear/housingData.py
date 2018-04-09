temp = []
x = []
y = []
with open('./housing_train.txt') as f:
	lines = f.read().split()
for count in range (len(lines)):
	if (count+1) % 14 == 0:
		y.append(lines[count])
	else:
		temp.append(lines[count])
x = [temp[i:i+13] for i in range(0,len(temp),13)]
print x