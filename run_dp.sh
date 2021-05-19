#!/bin/bash
for i in {1..5}
do
    python DBULL_dp.py \
        -bnpyPath $PATH_OF_bnpy \
        -outputPath $OUTPUT_PATH \
        -rootPath $PATH_OF_THIS_REPO \
        -epoch 10 \
        -Kmax 50 \
        -dataset mnist \
        -scale 0.05 \
        -batch_iter 3 \
        -logFile \
        -useLocal \
        -rep $i \
        -sf 0.1 \
        -learningRate 2e-4 \
        -batchsize 500 \
        -nBatch 1
done



