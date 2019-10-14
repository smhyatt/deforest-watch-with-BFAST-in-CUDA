CC=gcc
# CFLAGS= -std=c++17 -Wall -Werror -Wextra -pedantic
CFLAGS=

.PHONY: clean all
cmd:
	$(CC) $(CFLAGS) seq.c

insp:
	futhark c insp-data.fut
	./insp-data < data/peru.in > data/testset.in

gaussjordan: gaussjordan.c
	$(CC) $(CFLAGS) gaussjordan.c
