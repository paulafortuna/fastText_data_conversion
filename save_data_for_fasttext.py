# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import classifier_utils



def main():
  processors = {
      "cola": classifier_utils.ColaProcessor,
      "mnli": classifier_utils.MnliProcessor,
      "mismnli": classifier_utils.MisMnliProcessor,
      "mrpc": classifier_utils.MrpcProcessor,
      "rte": classifier_utils.RteProcessor,
      "sst-2": classifier_utils.Sst2Processor,
      "sts-b": classifier_utils.StsbProcessor,
      "qqp": classifier_utils.QqpProcessor,
      "qnli": classifier_utils.QnliProcessor,
      "wnli": classifier_utils.WnliProcessor,
      "offenseval": classifier_utils.Offen_1_Processor,
      "hateval_hate": classifier_utils.HatevalHate_Processor,
      "hateval_aggression": classifier_utils.HatevalAggression_Processor,
      "ami": classifier_utils.ami_Processor,
      "davidson_hate": classifier_utils.DavidsonHate_Processor,
      "davidson_offensive": classifier_utils.DavidsonOffensive_Processor,
      "davidson_toxicity": classifier_utils.DavidsonToxicity_Processor,
      "stormfront_post": classifier_utils.StormfrontPost_Processor,
      "stormfront_sentence": classifier_utils.StormfrontSentence_Processor,
      "toxicity_toxic": classifier_utils.ToxicityToxic_Processor,
      "toxicity_identityhate": classifier_utils.ToxicityIdentityHate_Processor,
      "toxicity_severetoxic": classifier_utils.ToxicitySevereToxic_Processor,
      "toxicity_insult": classifier_utils.ToxicityInsult_Processor,
      "toxicity_obscene": classifier_utils.ToxicityObscene_Processor,
      "toxicity_threat": classifier_utils.ToxicityThreat_Processor,
      "trac_covert": classifier_utils.TracCovert_Processor,
      "trac_overt": classifier_utils.TracOvert_Processor,
      "trac_aggr":  classifier_utils.TracAggr_Processor,
      "zeerak_hate": classifier_utils.ZeerakHate_Processor,
      "zeerak_racism": classifier_utils.ZeerakRacism_Processor,
      "zeerak_sexism": classifier_utils.ZeerakSexism_Processor,
      "founta_hateful": classifier_utils.FountaHateful_Processor,
      "founta_abusive": classifier_utils.FountaAbusive_Processor,
      "founta_toxicity": classifier_utils.FountaToxicity_Processor
  }

  processors_list = {
      #"trac_covert",
      #"trac_overt",
      #"trac_aggr",
      #"offenseval",
      #"hateval_hate",
      #"hateval_aggression",
      #"ami",
      #"davidson_hate",
      #"davidson_offensive",
      #"davidson_toxicity",
      #"stormfront_post",
      #"stormfront_sentence",
      #"toxicity_toxic",
      #"toxicity_identityhate",
      #"toxicity_severetoxic",
      #"toxicity_insult",
      #"toxicity_obscene",
      #"toxicity_threat",
      #"zeerak_hate",
      #"zeerak_racism",
      #"zeerak_sexism",
      "founta_hateful",
      "founta_abusive",
      "founta_toxicity"
  }

  data_dir = "training-albert"

  for task_name in processors_list:

      processor = processors[task_name](
          use_spm=False,
          do_lower_case=False)

      train_examples = processor.get_train_examples(data_dir)
      processor.write_txt(train_examples, "data_TACL/" + task_name + "_train.txt")
      predict_examples = processor.get_test_examples(data_dir)
      processor.write_txt(predict_examples, "data_TACL/" + task_name + "_test.txt")


if __name__ == "__main__":
    main()
