CC=gcc
# CFLAGS= -std=c++17 -Wall -Werror -Wextra -pedantic
CFLAGS=

.PHONY: clean all
cmd:
	$(CC) $(CFLAGS) seq.c && ./a.out
	rm a.out

rune:
	futhark c bfast-irreg.fut
	./bfast-irreg < data/testset_sahara_2pix.in > data/val.data
	$(CC) $(CFLAGS) seq.c && ./a.out >> data/val.data
	futhark c validation-benchmark.fut
	./validation-benchmark < data/val.data
	rm validation-benchmark
	rm bfast-irreg
	rm a.out

sarah:
	$(CC) $(CFLAGS) seq-parse.c 
	./a.out sahara
	rm a.out

peru:
	$(CC) $(CFLAGS) seq-parse.c 
	./a.out peru
	rm a.out

insp:
	futhark c insp-data.fut
	# ./insp-data < data/peru.in.gz > data/testset_peru_2pix.in
	./insp-data < data/sahara.in > data/testset_sahara_2pix.in
	rm insp-data insp-data.c

parseSahara:
	futhark opencl flatten.fut   && ./flatten   < data/sahara.in > data/sflat.in
	futhark opencl getCInput.fut && ./getCInput < data/sahara.in > data/sCInput.in
	python3 filterTypes.py sahara > data/saharaCimages.in
	python3 filterTypes.py sinput > data/saharaC.in
	rm flatten data/sflat.in data/sCInput.in getCInput

parsePeru:
	futhark opencl flatten.fut   && ./flatten   < data/peru.in > data/pflat.in
	futhark opencl getCInput.fut && ./getCInput < data/peru.in > data/pCInput.in
	python3 filterTypes.py peru   > data/peruCimages.in
	python3 filterTypes.py pinput > data/peruC.in
	rm data/pflat.in data/pCInput.in flatten getCInput

parse4C: parseSahara parsePeru

clean:
	rm a.out

decompress:
	gzip -k -d data/sahara.in
	gzip -k -d data/peru.in
