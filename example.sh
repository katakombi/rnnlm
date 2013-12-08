#!/bin/bash

#This is simple example how to use rnnlm tool for training and testing rnn-based language models

make clean
make

rm model
rm model.compressed
rm model.output.txt

#rnn model is trained here
time ./rnnlm -train train -valid valid -rnnlm model -hidden 15 -rand-seed 1 -debug 2 -class 100 -bptt 4 -bptt-block 10 -direct-order 3 -alpha 0.1 -direct 35
time ./rnnlm -rnnlm model -compress 4 -kmean 15 -write-compressed model.compressed -test test
./rnnlm -rnnlm model.compressed -test test
./rnnlm -rnnlm model -test test
