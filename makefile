CC = g++
WEIGHTTYPE = float
STRIDE = 8
CFLAGS = -D WEIGHTTYPE=$(WEIGHTTYPE) -D STRIDE=$(STRIDE) -lm -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result
#CFLAGS = -lm -O2 -Wall

all: rnnlmlib.o rnnlm

rnnlmlib.o : rnnlmlib.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlmlib.cpp

rnnlm : rnnlm.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlm.cpp rnnlmlib.o -o rnnlm

clean:
	rm -rf *.o rnnlm
