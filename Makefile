CC=gcc
# CFLAGS= -std=c++17 -Wall -Werror -Wextra -pedantic
CFLAGS=

.PHONY: clean all
cmd:
	$(CC) $(CFLAGS) seq.c

insp:
	futhark c insp-data.fut
	# ./insp-data < data/peru.in.gz > data/testset_peru_2pix.in
	./insp-data < data/sahara.in > data/testset_sahara_2pix.in

parseSahara:
	futhark opencl flatten.fut && ./flatten < data/sahara.in > data/sflat.in
	python3 filter.py sahara > data/saharaC.in
	rm data/sflat.in

parsePeru:
	futhark opencl flatten.fut && ./flatten < data/peru.in > data/pflat.in
	python3 filter.py peru > data/peruC.in
	rm data/pflat.in

parse4C: parseSahara parsePeru

gaussjordan: gaussjordan.c
	$(CC) $(CFLAGS) gaussjordan.c

clean:
	rm a.out

filtertypes:
	python3 filtertypes.py

decompress:
	gzip -k -d data/sahara.in
