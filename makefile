CC = g++
CFLAGS = -g -lm -O2 -Wall -funroll-loops -ffast-math -Wno-unused-result

all: rnnlmlib.o rnnlm

rnnlmlib.o : rnnlmlib.cpp rnnlmlib.h
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlmlib.cpp

rnnlm : rnnlm.cpp rnnlmlib.o
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlm.cpp rnnlmlib.o -o rnnlm

clean:
	rm -rf *.o rnnlm
