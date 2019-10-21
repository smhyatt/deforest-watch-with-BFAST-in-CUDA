#!/usr/bin/env python3

print('Filtering types information out for C/C++/CUDA input')

with open('data/testset_sahara_2pix.in') as f:
    read_data = f.read()

print(read_data)

filteredData = read_data.replace("i32", "").replace("f32", "")

print(filteredData)

