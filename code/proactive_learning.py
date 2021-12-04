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
"""


import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from modAL.models import ActiveLearner, Committee
from modAL.utils.selection import multi_argmax
from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy
from modAL.disagreement import max_disagreement_sampling
from modAL.utils.combination import make_linear_combination, make_product


# Set display options for pandas to show all the data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_item', None)
pd.set_option('display.max_rows', None)

# Set seed for the numpy random generator
np.random.seed(42)

# Number of parallel jobs to run.
# This setting is used by multiple libraries.
n_jobs = 6

# Number of cross-validation iterations to run while
# training various machine learning models in sklearn.
cv = 10


def calculate_utility_measure(classifier, instances):
    """
    Utility function to help rank instances so that the proactive learner
    can pick the most information instances for the next training iteration.

    :param classifier: Machine learning model to use to pick the most
                       informative instances for the next iteration of training.
    :param instances: The list of training instances in the pool from which the
                      next most informative instances will be picked.

    :return: A function that can calculate the utility of each instance in the pool
             and find the next most informative instances.
    """
    product1 = make_product(
        classifier_uncertainty, classifier_margin,
        exponents=[0.5, 0.5]
    )

    product2 = make_product(
        classifier_uncertainty, classifier_entropy,
        exponents=[0.5, 0.5]
    )

    product3 = make_product(
        classifier_margin, classifier_entropy,
        exponents=[0.5, 0.5]
    )

    linear_combination = make_linear_combination(
        product1, product2, product3,
        weights=[0.3, 0.3, 0.4]
    )

    return linear_combination(classifier, instances)


def instance_selector(utility, n_instances=1):
    """
    Function that selects the next most informative instances
    based on the utility function provided.

    :param utility: The utility function that calculate the amount
                    of information to be gained from all the instances
                    in the proactive learning training pool.
    :param n_instances: Number of informative instances to pick.

    :return: The most informative instances selected from the utility
             function, based on the classifier and list of training pool
             instances that were used to create the utility function.
    """
    return multi_argmax(utility, n_instances=n_instances)


def proactive_learner_query_strategy(classifier, instances, n_instances=1):
    """
    Custom proactive learning instance selection strategy. This query finds the utility
    of each instance in the pool and picks the most informative instances to use in the
    next iteration of training.

    :param classifier: Machine learning model to use to pick the most
                       informative instances for the next iteration of training.
    :param instances: The list of training instances in the pool from which the
                      next most informative instances will be picked.
    :param n_instances: Number of informative instances to pick.
    :return:
    """
    utility = calculate_utility_measure(classifier, instances)

    query_indexes = instance_selector(utility, n_instances=n_instances)

    return query_indexes


# Load the toy dataset generated using a RoBERTa model fine-tuned on emotion datasets
# The questions and responses have been split up
q_df = pd.read_csv('../data/q_emotion_predictions.csv')
r_df = pd.read_csv('../data/r_emotion_predictions.csv')

# Split the dataset into train, pool, validation, and test datasets
# Training dataset: Used as initial set of instances to trained the Proactive Learners
# Pool dataset: Used by the Proactive Learners to pick instances and train further
# Validation dataset: Used to compare the performance of the Proactive Learners after each iteration of learning
# Test dataset: Used to report performance of the final chosen model
q_X_train, q_X_test, q_y_train, q_y_test = train_test_split(
    q_df['text'],
    q_df['prediction'],
    test_size=0.2,
    random_state=42,
    stratify=q_df['prediction']
)

q_X_train.reset_index(drop=True, inplace=True)
q_X_test.reset_index(drop=True, inplace=True)
q_y_train.reset_index(drop=True, inplace=True)
q_y_test.reset_index(drop=True, inplace=True)

q_X_train, q_X_val, q_y_train, q_y_val = train_test_split(
    q_X_train,
    q_y_train,
    test_size=0.2,
    random_state=42,
    stratify=q_y_train
)

q_X_train.reset_index(drop=True, inplace=True)
q_X_val.reset_index(drop=True, inplace=True)
q_y_train.reset_index(drop=True, inplace=True)
q_y_val.reset_index(drop=True, inplace=True)

q_X_train, q_X_pool, q_y_train, q_y_pool = train_test_split(
    q_X_train,
    q_y_train,
    test_size=0.5,
    random_state=42,
    stratify=q_y_train
)

q_X_train.reset_index(drop=True, inplace=True)
q_X_pool.reset_index(drop=True, inplace=True)
q_y_train.reset_index(drop=True, inplace=True)
q_y_pool.reset_index(drop=True, inplace=True)

r_X_train, r_X_test, r_y_train, r_y_test = train_test_split(
    r_df['text'],
    r_df['prediction'],
    test_size=0.2,
    random_state=42,
    stratify=r_df['prediction']
)

r_X_train.reset_index(drop=True, inplace=True)
r_X_test.reset_index(drop=True, inplace=True)
r_y_train.reset_index(drop=True, inplace=True)
r_y_test.reset_index(drop=True, inplace=True)

r_X_train, r_X_val, r_y_train, r_y_val = train_test_split(
    r_X_train,
    r_y_train,
    test_size=0.2,
    random_state=42,
    stratify=r_y_train
)

r_X_train.reset_index(drop=True, inplace=True)
r_X_val.reset_index(drop=True, inplace=True)
r_y_train.reset_index(drop=True, inplace=True)
r_y_val.reset_index(drop=True, inplace=True)

r_X_train, r_X_pool, r_y_train, r_y_pool = train_test_split(
    r_X_train,
    r_y_train,
    test_size=0.5,
    random_state=42,
    stratify=r_y_train
)

r_X_train.reset_index(drop=True, inplace=True)
r_X_pool.reset_index(drop=True, inplace=True)
r_y_train.reset_index(drop=True, inplace=True)
r_y_pool.reset_index(drop=True, inplace=True)

# Setup three stacking classifiers as Proactive Learners in a committee.
# The parameters chosen were fine-tuned on a different dataset of text documents.
# Here, these parameters are used as the starting point and require further fine-tuning.
# TODO: Perform further fine-tuning using genetic search algorithms from tpot library.
q_estimators_stacking = [
    ('MultinomialNB', MultinomialNB(
        alpha=1e-05,
    )),
    ('KNeighborsClassifier', KNeighborsClassifier(
        n_neighbors=3,
        metric='cosine',
        n_jobs=n_jobs,
    )),
    ('RandomForestClassifier', RandomForestClassifier(
        n_estimators=125,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        n_jobs=n_jobs,
    )),
    ('SGDClassifier', SGDClassifier(
        alpha=1e-05,
        max_iter=500,
        tol=1e-4,
        learning_rate='adaptive',
        eta0=0.5,
        early_stopping=True,
        class_weight='balanced',
    )),
]

q_clf1 = StackingClassifier(
    estimators=q_estimators_stacking,
    final_estimator=GradientBoostingClassifier(
        subsample=0.75,
        min_samples_leaf=3,
        n_iter_no_change=5
    ),
    cv=cv,
    n_jobs=n_jobs,
)

q_clf2 = StackingClassifier(
    estimators=q_estimators_stacking,
    final_estimator=GradientBoostingClassifier(
        subsample=0.75,
        min_samples_leaf=3,
        n_iter_no_change=5
    ),
    cv=cv,
    n_jobs=n_jobs,
)

q_clf3 = StackingClassifier(
    estimators=q_estimators_stacking,
    final_estimator=GradientBoostingClassifier(
        subsample=0.75,
        min_samples_leaf=3,
        n_iter_no_change=5
    ),
    cv=cv,
    n_jobs=n_jobs,
)

# Each of the three Proactive Learners differs in their n-gram representations.
# q_estimator1 = Pipeline([
#     ('tfidf', TfidfVectorizer(
#         strip_accents='unicode',
#         stop_words='english',
#         ngram_range=(1, 1),
#         sublinear_tf=True
#     )),
#     ('clf', q_clf1),
# ])

# Use imblearn's pipeline to balance the dataset using SMOTE:
# Synthetic Minority Over-sampling Technique.
# 3 different proactive learners are setup, each using a different
# n-gram frequency in the TF-IDF vectorizer.
q_estimator1 = make_pipeline_imb(
    TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 1),
        sublinear_tf=True
    ),
    SMOTE(
        random_state=42,
        n_jobs=n_jobs
    ),
    q_clf1,
)

# q_estimator2 = Pipeline([
#     ('tfidf', TfidfVectorizer(
#         strip_accents='unicode',
#         stop_words='english',
#         ngram_range=(1, 2),
#         sublinear_tf=True
#     )),
#     ('clf', q_clf2),
# ])

q_estimator2 = make_pipeline_imb(
    TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True
    ),
    SMOTE(
        random_state=42,
        n_jobs=n_jobs
    ),
    q_clf2,
)

# q_estimator3 = Pipeline([
#     ('tfidf', TfidfVectorizer(
#         strip_accents='unicode',
#         stop_words='english',
#         ngram_range=(1, 3),
#         sublinear_tf=True
#     )),
#     ('clf', q_clf3),
# ])

q_estimator3 = make_pipeline_imb(
    TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 3),
        sublinear_tf=True
    ),
    SMOTE(
        random_state=42,
        n_jobs=n_jobs
    ),
    q_clf3,
)

q_learner1 = ActiveLearner(
    estimator=q_estimator1,
    query_strategy=proactive_learner_query_strategy,
    X_training=q_X_train, y_training=q_y_train,
)

q_learner2 = ActiveLearner(
    estimator=q_estimator2,
    query_strategy=proactive_learner_query_strategy,
    X_training=q_X_train, y_training=q_y_train,
)

q_learner3 = ActiveLearner(
    estimator=q_estimator3,
    query_strategy=proactive_learner_query_strategy,
    X_training=q_X_train, y_training=q_y_train,
)

# Create the final Proactive Learner as a committee of learners
q_learner_list = [
    q_learner1,
    q_learner2,
    q_learner3
]

# The Proactive Learner picks instances that the three underlying learners
# disagree on in terms of their confidence levels for the prediction.
q_learner = Committee(
    learner_list=q_learner_list,
    query_strategy=max_disagreement_sampling
)

# Repeat the same process to setup the Proactive Learners
# for the responses
r_estimators_stacking = [
    ('MultinomialNB', MultinomialNB(
        alpha=1e-05,
    )),
    ('KNeighborsClassifier', KNeighborsClassifier(
        n_neighbors=3,
        metric='cosine',
        n_jobs=n_jobs,
    )),
    ('RandomForestClassifier', RandomForestClassifier(
        n_estimators=125,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        n_jobs=n_jobs,
    )),
    ('SGDClassifier', SGDClassifier(
        alpha=1e-05,
        max_iter=500,
        tol=1e-4,
        learning_rate='adaptive',
        eta0=0.5,
        early_stopping=True,
        class_weight='balanced',
    )),
]

r_clf1 = StackingClassifier(
    estimators=r_estimators_stacking,
    final_estimator=GradientBoostingClassifier(
        subsample=0.75,
        min_samples_leaf=3,
        n_iter_no_change=5
    ),
    cv=cv,
    n_jobs=n_jobs,
)

r_clf2 = StackingClassifier(
    estimators=r_estimators_stacking,
    final_estimator=GradientBoostingClassifier(
        subsample=0.75,
        min_samples_leaf=3,
        n_iter_no_change=5
    ),
    cv=cv,
    n_jobs=n_jobs,
)

r_clf3 = StackingClassifier(
    estimators=r_estimators_stacking,
    final_estimator=GradientBoostingClassifier(
        subsample=0.75,
        min_samples_leaf=3,
        n_iter_no_change=5
    ),
    cv=cv,
    n_jobs=n_jobs,
)

r_estimator1 = Pipeline([
    ('tfidf', TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 1),
        sublinear_tf=True
    )),
    ('clf', r_clf1),
])

r_estimator2 = Pipeline([
    ('tfidf', TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('clf', r_clf2),
])

r_estimator3 = Pipeline([
    ('tfidf', TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 3),
        sublinear_tf=True
    )),
    ('clf', r_clf3),
])

r_learner1 = ActiveLearner(
    estimator=r_estimator1,
    query_strategy=proactive_learner_query_strategy,
    X_training=r_X_train, y_training=r_y_train,
)

r_learner2 = ActiveLearner(
    estimator=r_estimator2,
    query_strategy=proactive_learner_query_strategy,
    X_training=r_X_train, y_training=r_y_train,
)

r_learner3 = ActiveLearner(
    estimator=r_estimator3,
    query_strategy=proactive_learner_query_strategy,
    X_training=r_X_train, y_training=r_y_train,
)

r_learner_list = [
    r_learner1,
    r_learner2,
    r_learner3
]

r_learner = Committee(
    learner_list=r_learner_list,
    query_strategy=max_disagreement_sampling
)

# Calculate the performance of the Proactive Learners with each iteration of learning
q_performance_history = [f1_score(q_y_val, q_learner.predict(q_X_val), average='weighted')]
r_performance_history = [f1_score(r_y_val, r_learner.predict(r_X_val), average='weighted')]

# Proactive Learning parameters for number of instances to pick for each query
# and the number of total queries to perform.
n_queries = 10
n_instances = 25

# Proactive Learning main loop.
# Pick instances to learn and iteratively train the Machine Learning models.
for index in range(n_queries):
    # Pick the next most informative instances
    query_index, _ = q_learner.query(q_X_pool, n_instances=n_instances)

    # Train the Machine Learning models using the selected instance(s)
    q_learner.teach(
        X=q_X_pool.iloc[query_index],
        y=q_y_pool.iloc[query_index],
    )

    # Select instances without replacement
    q_X_pool.drop(index=query_index, inplace=True)
    q_y_pool.drop(index=query_index, inplace=True)

    # Reset the Pandas index after removing the instances
    q_X_pool.reset_index(drop=True, inplace=True)
    q_y_pool.reset_index(drop=True, inplace=True)

    # Store the performance of the model after the new learning phase
    # Use the validation dataset for this purpose.
    model_f1_score = f1_score(q_y_val, q_learner.predict(q_X_val), average='weighted')

    q_performance_history.append(model_f1_score)

    # Repeat all the steps for the responses
    query_index, _ = r_learner.query(r_X_pool, n_instances=n_instances)

    r_learner.teach(
        X=r_X_pool.iloc[query_index],
        y=r_y_pool.iloc[query_index],
    )

    r_X_pool.drop(index=query_index, inplace=True)
    r_y_pool.drop(index=query_index, inplace=True)

    r_X_pool.reset_index(drop=True, inplace=True)
    r_y_pool.reset_index(drop=True, inplace=True)

    model_f1_score = f1_score(r_y_val, r_learner.predict(r_X_val), average='weighted')

    r_performance_history.append(model_f1_score)


# Print the performance history on the validation set for the questions dataset
for index in range(len(q_performance_history)):
    print('q_learner weighted F1 score after query {n}: {f1:0.4f}'.format(n=index, f1=q_performance_history[index]))

pp = pprint.PrettyPrinter()

# Since the classes are imbalanced, print the performance metrics taking the imbalance into account
print('q_learner imbalanced classification report on validation dataset:')
pp.pprint(classification_report_imbalanced(q_y_val, q_learner.predict(q_X_val), output_dict=True))

# Repeat the process for the responses
for index in range(len(r_performance_history)):
    print('r_learner weighted F1 score after query {n}: {acc:0.4f}'.format(n=index, acc=r_performance_history[index]))

print('r_learner imbalanced classification report on validation dataset:')
pp.pprint(classification_report_imbalanced(r_y_val, r_learner.predict(r_X_val), output_dict=True))

# Plot the performance histories as a line graph
q_fig, q_ax = plt.subplots(figsize=(8.5, 6), dpi=130)

q_ax.plot(q_performance_history)
q_ax.scatter(range(len(q_performance_history)), q_performance_history, s=13)

q_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
q_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
q_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

q_ax.set_ylim(bottom=0, top=1)
q_ax.grid(True)

q_ax.set_title('Incremental weighted F1: q_learner')
q_ax.set_xlabel('Query iteration')
q_ax.set_ylabel('Weighted F1 score')

plt.show()

# Repeat the plot for the responses.
r_fig, r_ax = plt.subplots(figsize=(8.5, 6), dpi=130)

r_ax.plot(r_performance_history)
r_ax.scatter(range(len(r_performance_history)), r_performance_history, s=13)

r_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
r_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
r_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

r_ax.set_ylim(bottom=0, top=1)
r_ax.grid(True)

r_ax.set_title('Incremental weighted F1 score: r_learner')
r_ax.set_xlabel('Query iteration')
r_ax.set_ylabel('Weighted F1 score')

plt.show()
