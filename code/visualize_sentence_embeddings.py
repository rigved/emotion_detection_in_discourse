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

Code modified from https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, AutoConfig


# Set the numpy random seed for reproducibility
np.random.seed(42)

# Use the GPU if it's available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set the PyTorch seed for reproducibility
torch.cuda.manual_seed_all(42)


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


def get_sentence_vector(sentence, transformer_tokenizer, transformer_model):
    """
    Obtain a sentence embedding for the given sentence. The first '[CLS]' token
    of the sentence is considered to be a close-enough representation of the
    sentence.

    :param sentence: The sentence for which the embedding is required.
    :param transformer_tokenizer: The transformer tokenizer that can provide a tensor
                                  representation of the given input sentence.
    :param transformer_model: The transformer model that provides a prediction for the
                              given input sentence.

    :return: A sentence embedding for the given input sentence.
    """
    # Obtain a tensor representation of the sentence
    encoded = transformer_tokenizer.encode_plus(sentence, return_tensors="pt").to(device)

    # Run inference on the model using the input sentence tensor
    with torch.no_grad():
        output = transformer_model(**encoded)

    # The word embedding for the first token, i.e., [CLS]
    # encodes the meaning of the entire input sentence in
    # terms of the word embeddings.
    return np.array(output.hidden_states[0][0][0].cpu())


def fetch_sentence_embedding(text, text_type, label):
    """
    Obtain sentence embedding for the given sentence.

    :param text: The sentence for which the embedding is required.
    :param text_type: The non-emotion label attached to the sentence from the original dataset.
    :param label: The emotion label for the given input sentence, as a string.

    :return: The sentence embeddings for the given input sentence as a list.
    """
    return get_sentence_vector(text, tokenizer, model), text_type, label


# Load the dataset in the required format
df = pd.read_csv('../data/q_emotion_predictions.csv')
q_text_list = df['text'].tolist()
q_type_list = df['type'].tolist()
q_label_list = df['label'].tolist()

# Load the RoBERTa tokenizer and model that has been fine-tuned on various emotion datasets
# and create a trainer that can be used to fetch the sentence embedding.
model_name = 'j-hartmann/emotion-english-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure that the output includes the values of the hidden states
# so that we can fetch the sentence embedding from the hidden state.
config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
trainer = Trainer(model=model)

# Tokenize all the question sentences that we already split into individual sentences earlier
# and create a dataset loader from this data.
q_tokenized_texts = tokenizer(q_text_list, truncation=True, padding=True)
q_discourse_dataset = DiscourseDataset(q_tokenized_texts)

# Fetch the sentence embeddings
sentence_embeddings = list()

for index in range(len(q_text_list)):
    sentence_embeddings.append(fetch_sentence_embedding(
        q_text_list[index],
        q_type_list[index],
        q_label_list[index]
    ))

# Prepare the sentence embedding for t-SNE
sentence_embeddings_df = pd.DataFrame(
    sentence_embeddings,
    columns=['sentence_embedding', 'type', 'label']
)

# Perform t-SNE to reduce the embedding space to a 2-dimensional space
word_embeddings_2d = TSNE(n_components=2).fit_transform(
    sentence_embeddings_df['sentence_embedding'].tolist()
)

# Prepare the reduced vector space for plotting
word_embeddings_2d_df = pd.DataFrame(
    word_embeddings_2d,
    columns=['x', 'y']
)
word_embeddings_2d_df = word_embeddings_2d_df.join(sentence_embeddings_df['type'])
word_embeddings_2d_df = word_embeddings_2d_df.join(sentence_embeddings_df['label'])

# Select the colors of the various emotion scatter plot points
colors = ['red', 'chocolate', 'gold', 'limegreen', 'slategray', 'dodgerblue', 'blueviolet']
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Plot the scatter plot
fig, ax = plt.subplots()

for index in range(len(colors)):
    indexes = word_embeddings_2d_df['label'] == labels[index]

    ax.scatter(
        word_embeddings_2d_df.loc[indexes, 'x'],
        word_embeddings_2d_df.loc[indexes, 'y'],
        c=colors[index],
        label=labels[index]
    )

ax.legend(bbox_to_anchor=(2.0, 1.0), loc='upper right')
ax.grid(True)

plt.tight_layout()
plt.show()
