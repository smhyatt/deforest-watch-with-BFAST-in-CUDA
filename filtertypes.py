#!/usr/bin/env python3
import re


print('Filtering types information out for C/C++/CUDA input')

with open('data/testset_peru_2pix.in') as f:
    read_data = f.read()

print(read_data)

filteredData = re.search("i32", read_data)

print(filteredData)

