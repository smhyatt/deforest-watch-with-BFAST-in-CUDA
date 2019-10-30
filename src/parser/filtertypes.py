#!/usr/bin/env python3

import sys

if (sys.argv[1] == "peru"):
	with open('../data/pflat.in') as f:
		data = f.read()
elif (sys.argv[1] == "pinput"):
	with open('../data/pCInput.in') as f:
		data = f.read()
elif (sys.argv[1] == "sinput"):
	with open('../data/sCInput.in') as f:
		data = f.read()
elif (sys.argv[1] == "sahara"):
	with open('../data/sflat.in') as f:
		data = f.read()
elif (sys.argv[1] == "peru2"):
	with open('../data/p2flat.in') as f:
		data = f.read()
elif (sys.argv[1] == "pinput2"):
	with open('../data/p2CInput.in') as f:
		data = f.read()
elif (sys.argv[1] == "sinput2"):
	with open('../data/s2CInput.in') as f:
		data = f.read()
else: 
	with open('../data/s2flat.in') as f:
		data = f.read()


filteredData = data.replace("i32", "").replace("f32", "").replace("[", "").replace("]", "")
print(filteredData)
