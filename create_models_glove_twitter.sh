#!/usr/bin/env bash

## declare an array variable

declare -a names=("trac_covert" "trac_overt" "trac_aggr" "offenseval" "hateval_hate" "hateval_aggression" "ami" "davidson_hate" "davidson_offensive" "davidson_toxicity" "stormfront_post" "stormfront_sentence" "toxicity_toxic" "toxicity_identityhate" "toxicity_severetoxic" "toxicity_insult" "toxicity_obscene" "toxicity_threat" "zeerak_sexism" "zeerak_racism" "zeerak_hate")

for training in "${names[@]}"
do
# train pre trained twitter
./fastText/fasttext supervised -input ./data_TACL/train_data/${training}_train.txt -output ./models/model_pretrained_glove_twitter_${training} -epoch 25 -wordNgrams 2 -dim 200 -loss hs -thread 7 -minCount 1 -lr 1.0 -verbose 2 -pretrainedVectors ./fastText/glove.twitter.27B.200d.txt
done


## now loop through the above array
for training in "${names[@]}"
do
    for testing in "${names[@]}"
    do
    # test probabilities
    ./fastText/fasttext predict-prob ./models/model_pretrained_glove_twitter_${training}.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/pretrained_glove_twitter_${training}_predicts_${testing}.txt
    done
done
