CC=gcc
# CFLAGS= -std=c++17 -Wall -Werror -Wextra -pedantic
CFLAGS= -v

.PHONY: clean all
cmd:
	$(CC) $(CFLAGS) seq.c

insp:
	futhark c insp-data.fut
	./insp-data < data/peru.in > data/testset.in


