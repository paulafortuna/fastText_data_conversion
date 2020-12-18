#!/usr/bin/env bash

## declare an array variable

declare -a names=("trac_covert" "trac_overt" "trac_aggr" "offenseval" "hateval_hate" "hateval_aggression" "ami" "davidson_hate" "davidson_offensive" "davidson_toxicity" "stormfront_post" "stormfront_sentence" "toxicity_toxic" "toxicity_identityhate" "toxicity_severetoxic" "toxicity_insult" "toxicity_obscene" "toxicity_threat" "zeerak_sexism" "zeerak_racism" "zeerak_hate")

## now loop through the above array 5871
for training in "${names[@]}"
do
    for testing in "${names[@]}"
    do
    # train simple
    ./fastText/fasttext supervised -input ./data_TACL/train_data/${training}_train.txt -output ./models/model_simple_${training} -epoch 25 -wordNgrams 2 -dim 300 -loss hs -thread 7 -minCount 1 -lr 1.0 -verbose 2
    # test simple
    ./fastText/fasttext predict-prob ./models/model_simple_${training}.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/simple_${training}_predicts_${testing}.txt
    # train pre trained
    ./fastText/fasttext supervised -input ./data_TACL/train_data/${training}_train.txt -output ./models/model_pretrained_${training} -epoch 25 -wordNgrams 2 -dim 300 -loss hs -thread 7 -minCount 1 -lr 1.0 -verbose 2 -pretrainedVectors ./fastText/cc.en.300.vec
    # test probabilities
    ./fastText/fasttext predict-prob ./models/model_pretrained_${training}.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/pretrained_${training}_predicts_${testing}.txt
    done
done

./fastText/fasttext predict-prob ./models/model_pretrained_trac_overt.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/pretrained_trac_overt_predicts_${testing}.txt
./fastText/fasttext predict-prob ./models/model_pretrained_trac_overt.bin ./data_TACL/test_data/trac_overt_test.txt 1 > ./results_fasttext_models/pretrained_trac_overt_predicts_trac_overt.txt

./fastText/fasttext supervised -input ./data_TACL/train_data/founta_abusive_train.txt -output ./models/model_pretrained_founta_abusive -epoch 25 -wordNgrams 2 -dim 300 -loss hs -thread 7 -minCount 1 -lr 1.0 -verbose 2 -pretrainedVectors ./fastText/cc.en.300.vec
./fastText/fasttext supervised -input ./data_TACL/train_data/founta_hateful_train.txt -output ./models/model_pretrained_founta_hateful -epoch 25 -wordNgrams 2 -dim 300 -loss hs -thread 7 -minCount 1 -lr 1.0 -verbose 2 -pretrainedVectors ./fastText/cc.en.300.vec
./fastText/fasttext supervised -input ./data_TACL/train_data/founta_toxicity_train.txt -output ./models/model_pretrained_founta_toxicity -epoch 25 -wordNgrams 2 -dim 300 -loss hs -thread 7 -minCount 1 -lr 1.0 -verbose 2 -pretrainedVectors ./fastText/cc.en.300.vec

./fastText/fasttext predict-prob ./models/model_pretrained_founta_abusive.bin ./data_TACL/test_data/founta_abusive_test.txt 1 > ./results_fasttext_models/pretrained_founta_abusive_predicts_founta_abusive.txt
