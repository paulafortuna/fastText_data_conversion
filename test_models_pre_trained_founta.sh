#!/usr/bin/env bash

## declare an array variable

declare -a names=("trac_covert" "trac_overt" "trac_aggr" "offenseval" "hateval_hate" "hateval_aggression" "ami" "davidson_hate" "davidson_offensive" "davidson_toxicity" "stormfront_post" "stormfront_sentence" "toxicity_toxic" "toxicity_identityhate" "toxicity_severetoxic" "toxicity_insult" "toxicity_obscene" "toxicity_threat" "zeerak_sexism" "zeerak_racism" "zeerak_hate")



## now loop through the above array
for testing in "${names[@]}"
do
# test probabilities
./fastText/fasttext predict-prob ./models/model_pretrained_founta_abusive.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/pretrained_founta_abusive_predicts_${testing}.txt
./fastText/fasttext predict-prob ./models/model_pretrained_founta_hateful.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/pretrained_founta_hateful_predicts_${testing}.txt
./fastText/fasttext predict-prob ./models/model_pretrained_founta_toxicity.bin ./data_TACL/test_data/${testing}_test.txt 1 > ./results_fasttext_models/pretrained_founta_toxicity_predicts_${testing}.txt
done
