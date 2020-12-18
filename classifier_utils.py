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
"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import collections
import csv
import os
import pandas as pd


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def __init__(self, use_spm, do_lower_case):
    super(DataProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    lines = []
    df = pd.read_csv(input_file, sep='\t')
    for index, row in df.iterrows():
        lines.append(row.tolist())

    return lines

  @classmethod
  def write_txt(cls, examples, name):

      dfObj = pd.DataFrame(columns=['label', 'text'])
      for example in examples:
          mystr = ' '.join(example.text_a.splitlines())
          dfObj = dfObj.append({'label': "__label__" + str(example.label),
                                'text': mystr}, ignore_index=True)

      dfObj.to_csv(name,header=False,index=False,sep='\t')

      print("hello")


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "test_matched.tsv")),
        "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      # Note(mingdachen): We will rely on this guid for GLUE submission.
      guid = self.process_text(line[0])
      text_a = self.process_text(line[8])
      text_b = self.process_text(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MisMnliProcessor(MnliProcessor):
  """Processor for the Mismatched MultiNLI data set (GLUE version)."""

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "dev_mismatched.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MNLI", "test_mismatched.tsv")),
        "test")


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MRPC", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MRPC", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "MRPC", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text(line[3])
      text_b = self.process_text(line[4])
      if set_type == "test":
        guid = line[0]
        label = "0"
      else:
        label = self.process_text(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "CoLA", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "CoLA", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "CoLA", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        guid = line[0]
        text_a = self.process_text(line[1])
        label = "0"
      else:
        text_a = self.process_text(line[3])
        label = self.process_text(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class Sst2Processor(DataProcessor):
  """Processor for the SST-2 data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "SST-2", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "SST-2", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "SST-2", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      if set_type != "test":
        guid = "%s-%s" % (set_type, i)
        text_a = self.process_text(line[0])
        label = self.process_text(line[1])
      else:
        guid = self.process_text(line[0])
        # guid = "%s-%s" % (set_type, line[0])
        text_a = self.process_text(line[1])
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class StsbProcessor(DataProcessor):
  """Processor for the STS-B data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "STS-B", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "STS-B", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "STS-B", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [None]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[7])
      text_b = self.process_text(line[8])
      if set_type != "test":
        label = float(line[-1])
      else:
        label = 0
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class FountaHateful_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        ""
        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "founta/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "founta/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "founta/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] == "hateful":
                label = '1'
                print("founta")
            else:
                label = '0'
                print("no founta")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] == "hateful":
                label = '1'
                print("founta")
            else:
                label = '0'
                print("no founta")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FountaAbusive_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        ""
        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "founta/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "founta/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "founta/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] == "abusive":
                label = '1'
                print("founta")
            else:
                label = '0'
                print("no founta")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] == "abusive":
                label = '1'
                print("founta")
            else:
                label = '0'
                print("no founta")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FountaToxicity_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        ""
        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "founta/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "founta/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "founta/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] == "abusive":
                label = '1'
                print("founta")
            elif line[2] == "hateful":
                label = '1'
                print("founta")
            else:
                label = '0'
                print("no founta")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] == "abusive":
                label = '1'
                print("founta")
            elif line[2] == "hateful":
                label = '1'
                print("founta")
            else:
                label = '0'
                print("no founta")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Offen_1_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = line[0]
        text_a = line[0]
        if line[1] == "OFF":
          label = '1'
        else:
          label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples


class Stormfront_complete_Processor1(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_1.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples


class Stormfront_complete_Processor2(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_2.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor3(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_3.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor4(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_4.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor5(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_5.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor6(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_6.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor7(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_7.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor8(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_8.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor9(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_9.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor10(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_10.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor11(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_11.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class Stormfront_complete_Processor12(DataProcessor):
    def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_trn_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-training-v1.tsv")), "train")

    def get_dev_examples(self, data_dir):
      """See base class."""
      #        return self._create_test_examples(
      #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
      return self._create_dev_examples(
        self._read_tsv(os.path.join(data_dir, "one/offenseval-trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        lines = []

        dataset = pd.read_csv(os.path.join(data_dir, 'NaziData/dataset_without_nan_12.tsv'), sep='\t', encoding='utf-8', error_bad_lines=False)
        for index, row in dataset.iterrows():
            lines.append(row.tolist())

        return self._create_dev_examples(lines, "dev")

    def get_labels(self):
      """See base class."""
      return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        #            guid = "%s-%s" % (set_type, i)
        guid = line[0]
        text_a = line[1]
        if line[2] == "OFF":
          label = '1'
          print("OFF")
        else:
          label = '0'
          print("NO OFF")
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

    def _create_dev_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
        guid = i
        text_a = line[4]
        label = '0'
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples

class HatevalHate_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        ""
        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "hateval_hate/train_en.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "hateval_hate/dev_en.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "hateval_hate/dev_en.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] == 1:
                label = '1'
                print("train found hate")
            else:
                label = '0'
                print("train no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] == 1:
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class HatevalAggression_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        ""
        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "hateval_aggression/train_en.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "hateval_aggression/dev_en.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "hateval_aggression/dev_en.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[4] == 1:
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[4] == 1:
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class ami_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        ""
        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "ami/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "ami/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "ami/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[3] == 1:
                label = '1'
                print("misoginy")
            else:
                label = '0'
                print("no misoginy")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[3] == 1:
                label = '1'
                print("test found misoginy")
            else:
                label = '0'
                print("test no misoginy")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class DavidsonToxicity_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[7]
            if line[6] != 2:
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[7]
            if line[6] != 2:
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class DavidsonHate_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[7]
            if line[6] == 0:
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[7]
            if line[6] == 0:
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class DavidsonOffensive_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "davidson/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            print(line)
            guid = line[0]
            text_a = line[7]
            if line[6] == 1:
                label = '1'
                print("Davidson off")
            else:
                label = '0'
                print("Davidson no off")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[7]
            if line[6] == 1:
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class ToxicityToxic_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[3] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
                print("test found no toxic")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[3] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
                print("test no toxic")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ToxicityIdentityHate_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[8] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[8] == 1:
                label = '1'
                print("test found identity hate")
            else:
                label = '0'
                print("test no identity hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ToxicitySevereToxic_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[4] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[4] == 1:
                label = '1'
                print("test found severe")
            else:
                label = '0'
                print("test no severe")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ToxicityInsult_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[7] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[7] == 1:
                label = '1'
                print("test found severe")
            else:
                label = '0'
                print("test no severe")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ToxicityObscene_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[5] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[5] == 1:
                label = '1'
                print("test found severe")
            else:
                label = '0'
                print("test no severe")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ToxicityThreat_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "toxicity/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[6] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[6] == 1:
                label = '1'
                print("test found severe")
            else:
                label = '0'
                print("test no severe")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StormfrontPost_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "stormfront_post/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "stormfront_post/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "stormfront_post/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[3] == 1:
                label = '1'
                print("test found toxic")
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[3] == 1:
                label = '1'
                print("test found severe")
            else:
                label = '0'
                print("test no severe")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StormfrontSentence_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "stormfront_sentence/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "stormfront_sentence/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "stormfront_sentence/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[4]
            if line[5] == 'hate':
                label = '1'
                print("hate")
            else:
                label = '0'
                print("no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[4]
            if line[5] == 'hate':
                label = '1'
                print("test found severe")
            else:
                label = '0'
                print("test no severe")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class TracCovert_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[3] == "CAG":
                print("covert")
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[3] == "CAG":
                label = '1'
                print("test found CAG")
            else:
                label = '0'
                print("test no CAG")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class TracOvert_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[3] == "OAG":
                print("overt")
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[3] == "OAG":
                label = '1'
                print("test found OAG")
            else:
                label = '0'
                print("test no OAG")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class TracAggr_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "TRAC_aggressive_en/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        print("creating training examples")
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[2]
            if line[3] != "NAG":
                print("agg")
                label = '1'
            else:
                print("n agg")
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            if line[3] != "NAG":
                label = '1'
                print("test found OAG")
            else:
                label = '0'
                print("test no OAG")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ZeerakHate_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        examples = self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/test.tsv")), "dev")
        print("**************************************************")
        print("**************************************************")
        print("**************************************************")
        print("**************************************************")
        print("**************************************************")
        print("**************************************************")
        for example in examples:
            print(example.guid)
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] != "neither":
                print("hate")
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] != "neither":
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ZeerakRacism_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] == "racism":
                print("hate")
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] == "racism":
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ZeerakSexism_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_trn_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        #        return self._create_test_examples(
        #            self._read_tsv(os.path.join(data_dir, "testset-taska.tsv")), "dev")
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "zeerak/test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_trn_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        print("creating training examples")
        examples = []
        for (i, line) in enumerate(lines):
            #            guid = "%s-%s" % (set_type, i)
            guid = line[0]
            text_a = line[1]
            if line[2] == "sexism":
                print("hate")
                label = '1'
            else:
                label = '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            if line[2] == "sexism":
                label = '1'
                print("test found hate")
            else:
                label = '0'
                print("test no hate")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QqpProcessor(DataProcessor):
  """Processor for the QQP data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QQP", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QQP", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QQP", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = line[0]
      # guid = "%s-%s" % (set_type, line[0])
      if set_type != "test":
        try:
          text_a = self.process_text(line[3])
          text_b = self.process_text(line[4])
          label = self.process_text(line[5])
        except IndexError:
          continue
      else:
        text_a = self.process_text(line[1])
        text_b = self.process_text(line[2])
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class QnliProcessor(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QNLI", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QNLI", "dev.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "QNLI", "test.tsv")),
        "test_matched")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type == "test_matched":
        label = "entailment"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class RteProcessor(DataProcessor):
  """Processor for the RTE data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "RTE", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "RTE", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "RTE", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class WnliProcessor(DataProcessor):
  """Processor for the WNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "WNLI", "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "WNLI", "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "WNLI", "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = self.process_text(line[0])
      # guid = "%s-%s" % (set_type, line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type != "test":
        label = self.process_text(line[-1])
      else:
        label = "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class AXProcessor(DataProcessor):
  """Processor for the AX data set (GLUE version)."""

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "diagnostic", "diagnostic.tsv")),
        "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      # Note(mingdachen): We will rely on this guid for GLUE submission.
      guid = self.process_text(line[0])
      text_a = self.process_text(line[1])
      text_b = self.process_text(line[2])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_name):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  if task_name != "sts-b":
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  print(example.text_a)
  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  # The convention in ALBERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if task_name != "sts-b":
    label_id = label_map[example.label]
  else:
    label_id = example.label

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature
