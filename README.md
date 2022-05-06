# Introduction: Emotion detection in U.S. Congressional Hearings

In this project, we annotate an existing dataset of U. S. Congressional Hearings (Ferracane et al., 2021) with emotions and the phrases where these emotions are indicated. We analyse these annotations to find meaningful patterns. Furthermore, we build a classifier solution to predict these emotions labels and show that Proactive Learning (Donmez and Carbonell, 2008; Ambati and Carbonell, 2009) can reduce training costs for this classifier.


# Installation

Run the following command to install the dependencies. It is recommended that you setup
a Python virtual environment for this project.

```commandline
pip install -r requirements.txt
```

If you would like to generate the documentation, then please install `doxygen` and `doxypy`
via your system's package manager.


# Data

The data is available in the `data` folder. It contains the original
20 U.S. Congressional Hearings data from Ferracane et al. (2021) as the file
`expanded_with_features_annotated_questions_responses_gold.csv`. The emotion
labels from the current annotations is saved in the file
`emotion_annotated_data.csv`.


# Run

Run the code as follows:

```commandline
cd code
python proactive_learning.py
```

`proactive_learning.py`: This emulates the Proactive Learning technique proposed by
                         Donmez and Carbonell, 2008. It uses the modAL (Danka and Horvath, 2018)
                         Python library as the Active Learning framework and builds on top of that.
                         NOTE: This code is still a work-in-progress and doesn't completely 
                         cover all aspects of the Proactive Learning technique yet.

On the other hand, if you would like to run the exploratory data analysis Jupyter notebook, then
run the following command:

```commandline
cd code
jupyter notebook Exploratory_Data_Analysis.ipynb
jupyter notebook Exploratory_Data_Analysis_Survey.ipynb
```

Once the Jupyter notebook is open, run `Kernel -> Restart & Run All` to re-run all the analysis.
This analysis ia a work-in-progress as we complete the various parts of the project.

If you would like to generate the documentation, then run the following command:

```commandline
./generate_documentation.sh
```


# Architecture

![Architecture of the underlying Proactive Learner](Architecture_Diagram_Proactive_Learner.png 'Proactive Learner Architecture')

The classifier provides a Scikit-Learn (Pedregosa et al., 2011) API. This allows the classifier to be used for fine-tuning using available libraries. It also allows us to use this classifier within the modAL (Danka and Horvath, 2018) modular Active Learning framework.


# Experimental Design

![Experimental Design Methodology for the Proactive Learning experiment](Experiment_Design_Diagram_Proactive_Learner.png 'Experimental Design Methodology for the Proactive Learning experiment')

Donmez and Carbonell (2008) introduce the Proactive Learning technique that allows for many practical considerations in the traditional Active Learning paradigm. Some of these are:

* There are multiple oracles, each with a different level of expertise. The oracles’ sample-weighted F1-score on the validation dataset is taken as their reliability.
* The oracles might have different query costs. A higher cost will be attached to the labels obtained from more reliable oracles than from less reliable ones. Again, the sample-weighted F1-score on the validation dataset is taken as the query cost for a single label.


# Results

![Results from the Proactive Learning experiment](Results.png 'Results')

The above figure shows the trade-off between query cost and training performance. When the validation performance in the training phase starts to drop, the Proactive Learner selects a new oracle with a higher reliability. This new oracle incurs a higher query cost than the previous oracle. As soon as the validation performance in training starts improving, the previous (less reliable and less costly) oracle
is chosen again, ensuring optimal training at least cost. This technique performs identical to at least three baseline classifiers that use Entropy Sampling active learning selection criteria, Margin Sampling selection criteria, and Uncertainty Sampling selection criteria. Thus, Proactive Learning is a viable Active Learning technique.

![Correlation between the emotion intensity and the annotator’s confidence in that emotion being present in the given data instance. Spearman rank-correlation coefficient = 0.78.](Emotion_Intensity_Annotator_Confidence_Correlation.png 'Correlation')

In the above figure, we see a positive correlation in the distribution and the medians of the confidence level per intensity, with a Spearman rank-correlation co-efficient of 0.78.


# References

E. Ferracane, G. Durrett, J.J. Li, and K. Erk. 2021. Did they answer? subjective acts and intents in conversational discourse. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1626–1644, Online. Association for Computational Linguistics.

P. Donmez and J.G. Carbonell. 2008. Proactive learning: Cost-sensitive active learning with multiple imperfect oracles. In Proceedings of the 17th ACM Conference on Information and Knowledge Management, CIKM'08, page 619–628, New York, NY, USA. Association for Computing Machinery.

T. Danka and P. Horvath. 2018. modAL: A modular active learning framework for Python. Available on arXiv at https://arxiv.org/abs/1805.00979.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.


# Acknowledgements

We would like to thank [Research Computing at the Rochester Institute of Technology](https://doi.org/10.34788/0S3G-QD15) for providing the computing resources for the project.
