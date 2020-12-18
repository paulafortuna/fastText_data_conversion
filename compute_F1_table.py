import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

df_result = 0


processors_list = {"trac_covert",
      "trac_overt",
      "trac_aggr",
      "offenseval",
      "hateval_hate",
      "hateval_aggression",
      "ami",
      "davidson_hate",
      "davidson_offensive",
      "davidson_toxicity",
      "stormfront_post",
      "stormfront_sentence",
      "toxicity_toxic",
      "toxicity_identityhate",
      "toxicity_severetoxic",
      "toxicity_insult",
      "toxicity_obscene",
      "toxicity_threat",
      "zeerak_hate",
      "zeerak_racism",
      "zeerak_sexism",
      "founta_hateful",
      "founta_abusive",
      "founta_toxicity"
  }

model_list = {
    "pretrained",
    #"simple",
    #"pretrained_glove_twitter"
}

column_names = ["fasttext_model", "dataset", "model", "F1_macro"]
res_df = pd.DataFrame(columns = column_names)
for model in model_list:
    for training_dataset in processors_list:
        for testing_dataset in processors_list:
            try:
                load_path = "./results_metrics/F1_" + model + "_" + training_dataset + "_predicts_" + testing_dataset + ".csv"
                print(load_path)
                df = pd.read_csv(load_path)

                res_df = res_df.append({'fasttext_model': model,
                                        'dataset': testing_dataset,
                                        'model': training_dataset,
                                        'F1_macro': df['f1'][1]}, ignore_index=True)
                print(res_df)
            except:
                pass

res_df.to_csv("./F1_table.csv", index = False)
