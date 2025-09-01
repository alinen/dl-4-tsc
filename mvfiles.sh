#! /usr/bin/bash

for f in *_TRAIN.tsv; do echo ${f%%_TRAIN.tsv}; mkdir -p ${f%%_TRAIN.tsv}; done
for f in *_TRAIN.tsv; do mv $f ${f%%_TRAIN.tsv}; done
for f in *_TEST.tsv; do mv $f ${f%%_TEST.tsv}; done
