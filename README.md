# next_sentence_prediction_using_BERT

This project demonstrates how to train a BERT model for Next Sentence Prediction (NSP) using the Hugging Face Transformers library.

# Overview
Next Sentence Prediction is a binary classification task where a model predicts whether a second sentence logically follows a first sentence. This task is crucial for BERT's understanding of sentence relationships and is part of its pre-training objective.

# Project Structure
The project is implemented as a Jupyter Notebook and covers the following steps:

Setup and Installation: Installs necessary libraries like transformers, torch, datasets, matplotlib, and scikit-learn.

Dataset Creation: Defines a small, synthetic dataset of sentence pairs with labels indicating whether the second sentence is a "next sentence" (1) or "not next" (0). This dataset is then split into training and testing sets.

Tokenization: Utilizes the BertTokenizer to tokenize the sentence pairs, adding special tokens, truncation, and padding to prepare the data for BERT.

Model Loading and Training Arguments: Loads a pre-trained BertForNextSentencePrediction model and defines the TrainingArguments for the training process, including learning rate, batch sizes, number of epochs, and logging settings.

Training and Evaluation:

Defines a compute_metrics function to calculate accuracy during evaluation.

Initializes and runs the Trainer to fine-tune the BERT model on the prepared dataset.

Evaluates the trained model on the test set and prints the evaluation results, including accuracy.

Prediction Probabilities Visualization: Predicts probabilities for the test dataset and visualizes the "Next Sentence" vs. "Not Next" probabilities for a few sample predictions using bar plots.

# Requirements
To run this notebook, you will need the following Python libraries:

transformers

torch

datasets

matplotlib

scikit-learn


# Visualizations
The notebook includes bar charts to visualize the predicted probabilities for "Next Sentence" and "Not Next" for a few samples from the test set, along with their true labels. This helps in understanding the model's confidence in its predictions.
