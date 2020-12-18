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
for model in model_list:
    for training_dataset in processors_list:
        for testing_dataset in processors_list:
            try:
                print("training_dataset")
                print(training_dataset)
                print("testing_dataset")
                print(testing_dataset)
                # read testing original label
                path_testing_dataset = "./data_TACL/test_data/" + testing_dataset + "_test.txt"
                df_testing_dataset = pd.read_csv(path_testing_dataset, sep='\t', header=None)
                # read training_testing prediction
                path_training_testing_prediction = "./results_fasttext_models/" + model + "_" + training_dataset + "_predicts_" + testing_dataset + ".txt"
                print(path_training_testing_prediction)
                df_training_testing_prediction = pd.read_csv(path_training_testing_prediction, sep=' ', header=None)

                df_testing_dataset[0] = df_testing_dataset[0].str.replace(r'__label__', '')
                df_training_testing_prediction[0] = df_training_testing_prediction[0].str.replace(r'__label__', '')


                df_F1 = pd.DataFrame()
                df_F1 = df_F1.append(pd.Series(np.asarray(precision_recall_fscore_support(pd.to_numeric(df_testing_dataset[0]),pd.to_numeric(df_training_testing_prediction[0]),pos_label=1, average='binary'))),ignore_index=True)
                df_F1 = df_F1.append(pd.Series(np.asarray(precision_recall_fscore_support(pd.to_numeric(df_testing_dataset[0]),pd.to_numeric(df_training_testing_prediction[0]), average='macro'))),ignore_index=True)
                df_F1 = df_F1.append(pd.Series(np.asarray(precision_recall_fscore_support(pd.to_numeric(df_testing_dataset[0]),pd.to_numeric(df_training_testing_prediction[0]), average='micro'))),ignore_index=True)
                df_F1 = df_F1.append(pd.Series(np.asarray(precision_recall_fscore_support(pd.to_numeric(df_testing_dataset[0]),pd.to_numeric(df_training_testing_prediction[0]), average='weighted'))),ignore_index=True)

                df_F1.columns = ['precision', 'recall', 'f1', 'f1_type']
                df_F1['f1_type'] = ['binary','macro','micro','weighted']

                save_path = "./results_metrics/F1_" + model + "_" + training_dataset + "_predicts_" + testing_dataset + ".csv"
                df_F1.to_csv(save_path, index = False)
            except:
                pass
