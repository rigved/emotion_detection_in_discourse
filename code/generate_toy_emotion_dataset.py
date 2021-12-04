"""
What triggered that emotion? Emotion cause extraction in conversational discourse.
Copyright (C) 2021  Rigved Rakshit

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Code modified from https://colab.research.google.com/github/j-hartmann/emotion-english-distilroberta-base/blob/main/emotion_prediction_example.ipynb
"""

import os
import pandas as pd
import numpy as np
import nltk
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


# Disable parallelism in the tokenizers because it does not support forking the parent
# Python process after the tokenizers have already been run at least once.
# This does not cause any performance issues because the dataset is very small.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Maximum number of parallel jobs to run.
# It is recommended to set it to one less than the number of CPUs on the system.
max_workers = 11


class DiscourseDataset(Dataset):
    """
    Convert the data into a dataset that works as a data loader for the transformer model.
    """
    def __init__(self, tokenized_texts):
        """
        Initialize the dataset.

        :param tokenized_texts: The texts tokenized by an appropriate transformer tokenizer.
        """
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        """
        Calculate the total number of text documents.

        :return: The total number of text documents.
        """
        return len(self.tokenized_texts['input_ids'])

    def __getitem__(self, idx):
        """
        Allow Python-style [] indexing.

        :param idx: Index of the item to retrieve.

        :return: The requested item from the list of tokenized texts.
        """
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


# Load the dataset
df = pd.read_csv('../data/expanded_with_features_annotated_questions_responses_gold.csv')
df.drop_duplicates(subset='qa_index', inplace=True, ignore_index=True)


def split_sentences(text, text_type):
    """
    Split each document into individual sentences so that we can find the emotion conveyed in each sentence.

    :param text: The document containing multiple sentences.
    :param text_type: The label attached with the document so that we can replicate the label for each sentence.

    :return: A list of tuples containing the sentences and their corresponding labels.
    """
    sentence_list = nltk.sent_tokenize(text)
    return list(zip(sentence_list, [text_type] * len(sentence_list)))


# Split the each question and response into individual sentences
# and replicate the label to each of those sentences in parallel.
q_text_list = df['q_text'].tolist()
q_type_list = df['gold_q_intent'].tolist()
r_text_list = df['r_text'].tolist()
r_type_list = df['gold_coarse_label'].tolist()

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(split_sentences, q_text_list, q_type_list))

q_text_list = sum(results, [])
q_text_df = pd.DataFrame(q_text_list, columns=['text', 'type'])

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(split_sentences, r_text_list, r_type_list))

r_text_list = sum(results, [])
r_text_df = pd.DataFrame(r_text_list, columns=['text', 'type'])

# Use the GPU if it's available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the PyTorch seed for reproducibility
torch.cuda.manual_seed_all(42)

# Load the RoBERTa tokenizer and model that has been fine-tuned on various emotion datasets
# and create a trainer that can be used for inference.
model_name = 'j-hartmann/emotion-english-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
trainer = Trainer(model=model)

# Tokenize all the question sentences that we already split into individual sentences earlier
# and create a dataset loader from this data.
q_tokenized_texts = tokenizer(q_text_df['text'].tolist(), truncation=True, padding=True)
q_discourse_dataset = DiscourseDataset(q_tokenized_texts)

# Store the emotion predictions on the question dataset where we are only interested in the
# majority emotion.
# TODO: Use multiple emotion that are above a threshold so this becomes a multi-label problem.
q_predictions = trainer.predict(q_discourse_dataset)
q_majority_predictions = q_predictions.predictions.argmax(-1)
q_labels = pd.Series(q_majority_predictions).map(model.config.id2label)
q_scores = (np.exp(q_predictions[0]) / np.exp(q_predictions[0]).sum(-1, keepdims=True)).max(1)
q_scores_raw = (np.exp(q_predictions[0]) / np.exp(q_predictions[0]).sum(-1, keepdims=True))

# Repeat this process for the response dataset
r_tokenized_texts = tokenizer(r_text_df['text'].tolist(), truncation=True, padding=True)
r_discourse_dataset = DiscourseDataset(r_tokenized_texts)

r_predictions = trainer.predict(r_discourse_dataset)
r_majority_predictions = r_predictions.predictions.argmax(-1)
r_labels = pd.Series(r_majority_predictions).map(model.config.id2label)
r_scores = (np.exp(r_predictions[0]) / np.exp(r_predictions[0]).sum(-1, keepdims=True)).max(1)
r_scores_raw = (np.exp(r_predictions[0]) / np.exp(r_predictions[0]).sum(-1, keepdims=True))


def generate_csv(
        text_list,
        text_labels,
        majority_predictions,
        labels,
        scores,
        scores_raw,
        csv_file_path
):
    """
    Store the predicted emotions in a CSV file for later use.

    :param text_list: The list of sentences from the text for which the emotion labels are required.
    :param text_labels: The list of non-emotion lables (like question-types or response-labels) from the original dataset.
    :param majority_predictions: The list of emotion labels for the corresponding list of sentences, as integers.
    :param labels: The list of emotion labels for the corresponding list of sentences, as strings.
    :param scores: The prediction probability score for the majority emotion.
    :param scores_raw: The prediction probability scores for the individual emotion labels for each sentence.
    :param csv_file_path: The path to the CSV file where we want to store the data.
    """
    anger = []
    disgust = []
    fear = []
    joy = []
    neutral = []
    sadness = []
    surprise = []

    for index in range(len(text_list)):
        anger.append(scores_raw[index][0])
        disgust.append(scores_raw[index][1])
        fear.append(scores_raw[index][2])
        joy.append(scores_raw[index][3])
        neutral.append(scores_raw[index][4])
        sadness.append(scores_raw[index][5])
        surprise.append(scores_raw[index][6])

    df_predictions = pd.DataFrame(list(
        zip(text_list, text_labels, majority_predictions, labels, scores, anger, disgust, fear, joy, neutral, sadness, surprise)),
        columns=['text', 'type', 'prediction', 'label', 'score', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    )
    df_predictions.to_csv(path_or_buf=csv_file_path, index=False)


# Number of CSV files that we want to generate in parallel
max_workers = 2

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(
        generate_csv,
        [q_text_df['text'].tolist(), r_text_df['text'].tolist()],
        [q_text_df['type'].tolist(), r_text_df['type'].tolist()],
        [q_majority_predictions, r_majority_predictions],
        [q_labels, r_labels],
        [q_scores, r_scores],
        [q_scores_raw, r_scores_raw],
        ['../data/q_emotion_predictions.csv', '../data/r_emotion_predictions.csv']
    ))
