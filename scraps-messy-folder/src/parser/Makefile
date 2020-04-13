# PARSER FOLDER
# MAKEFILE FOR PARSING ALL DATA IN 

# Creating a testset of 2 pixels for both Sahara and Peru, to run bfast-irreg.fut

bfast2pixel: 
	futhark opencl bfast-irreg-testset-creator.fut
	./bfast-irreg-testset-creator < ../data/sahara.in > ../data/sahara2pix.in
	./bfast-irreg-testset-creator < ../data/peru.in   > ../data/peru2pix.in


# Parsing the full dataset of Sahara and Peru to use in the C++ and CUDA implementations

fullSahara:
	futhark opencl flatten.fut   && ./flatten   < ../data/sahara.in > ../data/sflat.in
	futhark opencl getCInput.fut && ./getCInput < ../data/sahara.in > ../data/sCInput.in
	python3 filtertypes.py sahara > ../data/saharaCimages.in
	python3 filtertypes.py sinput > ../data/saharaC.in
	rm flatten ../data/sflat.in ../data/sCInput.in getCInput || true

fullPeru:
	futhark opencl flatten.fut   && ./flatten   < ../data/peru.in > ../data/pflat.in
	futhark opencl getCInput.fut && ./getCInput < ../data/peru.in > ../data/pCInput.in
	python3 filtertypes.py peru   > ../data/peruCimages.in
	python3 filtertypes.py pinput > ../data/peruC.in
	rm ../data/pflat.in ../data/pCInput.in flatten getCInput || true



# Parsing the a 2 pixel dataset of Sahara and Peru to use in the C++ and CUDA implementations

2Sahara:
	futhark opencl flatten-testset.fut   && ./flatten-testset    < ../data/sahara.in > ../data/s2flat.in
	futhark opencl getCInput.fut && ./getCInput < ../data/sahara.in > ../data/s2CInput.in
	python3 filtertypes.py sahara2 > ../data/sahara2Cimages.in
	python3 filtertypes.py sinput2 > ../data/sahara2C.in
	rm flatten-testset ../data/s2flat.in ../data/s2CInput.in getCInput || true

2Peru:
	futhark opencl flatten-testset.fut   && ./flatten-testset   < ../data/peru.in > ../data/p2flat.in
	futhark opencl getCInput.fut && ./getCInput < ../data/peru.in  > ../data/p2CInput.in
	python3 filtertypes.py peru2   > ../data/peru2Cimages.in
	python3 filtertypes.py pinput2 > ../data/peru2C.in
	rm ../data/p2flat.in ../data/p2CInput.in flatten-testset getCInput || true


# Command to parse both testsets at once 

parsetestsets: bfast2pixel 2Sahara 2Peru


# Command to parse full datasets at once 

parse: parsetestsets fullSahara fullPeru


# Command to parse everything at once 

parse: parsetestsets fullSahara fullPeru


# Decompressing the data

decompress:
	gzip -d < ../data/peru.in.gz   > ../data/peru.in
	gzip -d < ../data/sahara.in.gz > ../data/sahara.in






