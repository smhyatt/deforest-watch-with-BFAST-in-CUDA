#!/usr/bin/env python3

import sys

if (sys.argv[1] == "peru"):
	with open('data/pflat.in') as f:
		data = f.read()
else:
	with open('data/sflat.in') as f:
		data = f.read()

filteredData = data.replace("i32", "").replace("f32", "").replace("[", "").replace("]", "")
print(filteredData)
