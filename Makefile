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

gaussjordan: gaussjordan.c
	$(CC) $(CFLAGS) gaussjordan.c

clean:
	rm a.out

filtertypes:
	python3 filtertypes.py

decompress:
	gzip -k -d data/sahara.in
