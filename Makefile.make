CC=gcc
CFLAGS=-std=c11 -Wall -Werror -Wextra -pedantic -g

.PHONY: clean all

cmd: 
	$(CC) $(CFLAGS) -o out -c seq.cpp

