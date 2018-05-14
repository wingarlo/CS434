import os
import subprocess

learn_rate = ["0.1","0.01","0.001","0.0001"]
weight_decay = ["0.", "0.001", "0.01", "0.1", "0.3", "0.7"]
drop_out = ["0.9", "0.5", "0.2", "0.1", "0.01"]
momentum = ["0.", "0.25","0.5" "0.75" , "0.9"]

max_acc = 0
for a in learn_rate:
	b = "0."
	c = "0.2"
	d = "0.5"
	max_acc = subprocess.check_output('python 3draft.py ' + a + " " + b + " "  + c + " " + d)
print max_acc
'''
for b in weight_decay:
	a = "0.1"
	c = "0.2"
	d = "0.5"
	os.system('python drossdraft.py ' + a + " " + b + " "  + c + " " + d)
for c in drop_out:
	a = "0.1"
	b = "0."
	d = "0.5"
	os.system('python drossdraft.py ' + a + " " + b + " "  + c + " " + d)	
for d in momentum:
	a = "0.1"
	b = "0."
	c = "0.2"
	os.system('python drossdraft.py ' + a + " " + b + " "  + c + " " + d)

for a in learn_rate:
	for b in weight_decay:
		for c in drop_out:
			for d in momentum:
				os.system('python 3draft.py ' + a + " " + b + " "  + c + " " + d)
				
'''