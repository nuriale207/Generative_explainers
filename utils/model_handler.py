import argparse

import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM, AutoConfig,AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from datasets import Dataset
from transformers import TrainingArguments

import torch

def load_model_multi_label(model_path, labels2id, id2label,load_tokenizer=True):
    """
    Load a model for multi-label classification
    :param model_path: path to the model
    :param labels2id: labels to id dictionary
    :param id2label: id to label dictionary
    :param load_tokenizer: weather to load the tokenizer or not
    :return:
        tokenizer: the tokenizer
        model: the model
        config: the model configuration
    """
    device = "cuda"  # the device to load the model onto
    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer=None
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=len(labels2id),
                                                                   problem_type="multi_label_classification",
                                                                   id2label=id2label,
                                                                   label2id=labels2id,
                                                               ignore_mismatched_sizes=True)

    return tokenizer,model,config

def load_model(model_path,from_scratch=False):
    """
    Load a model for masked language modeling
    :param model_path: path to the model
    :param from_scratch: weather to load the model from scratch or not
    :return: tokenizer, model, config
    """
    device = "cuda"  # the device to load the model onto
    if from_scratch:
        tokenizer=None
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_config(config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)

    return tokenizer,model,config




def load_tokenizer(tokenizer_path,tokenizer_id=None):
    """
    Load a tokenizer from a file or from the huggingface model hub
    :param tokenizer_path: path to the tokenizer file
    :param tokenizer_id: huggeface model id
    :return: tokenizer
    """
    if tokenizer_id is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    else:
        # Load the trained tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)

        # Convert the tokenizers library tokenizer to a format compatible with transformers
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        tokenizer = fast_tokenizer
    return tokenizer

def train_tokenizer(data,save_path="tokenizer_train_data.txt", save=True):
    """
    Train a tokenizer from a dataset
    :param data: the dataset to train the tokenizer
    :param save_path: the path to save the tokenizer
    :param save: weather to save the tokenizer or not
    :return: tokenizer
    """
    codes = data["PretrainSequence"]
    all_codes = list(data["PretrainSequence"])

    all_codes = [item.split(" ") for item in all_codes]

    all_codes_counts = [item for sublist in all_codes for item in sublist]

    all_codes_counts = list(set(all_codes_counts))

    file_name = "tokenizer_train_data.txt"

    # Open the file for writing
    with open(file_name, "w") as file:
        # Iterate over each inner list
        for row in all_codes:
            # Convert each element to string, join with a space, and write to the file
            file.write(" ".join(map(str, row)) + "\n")

    # Initialize a tokenizer with the WordPiece model
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # We add the ICD codes to the vocabulary so that they are not splitted
    tokenizer.add_tokens(all_codes_counts)

    # Initialize a pre-tokenizer to split the input into words
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize the trainer with your desired settings
    trainer = WordPieceTrainer(
        vocab_size=len(all_codes_counts),  # You can adjust the vocabulary size
        min_frequency=1,  # Minimum frequency for a token to be included in the vocabulary
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
        continuing_subword_prefix="##",  # Important for BERT-like tokenization
    )

    # List of files for training the tokenizer
    files = ["tokenizer_train_data.txt"]  # Replace with your file paths

    # Train the tokenizer
    tokenizer.train(files, trainer)
    if save:
        # Save the tokenizer to use later
        tokenizer.save(save_path)

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    tokenizer = fast_tokenizer

    return tokenizer




def add_codes_tokenizer(model,tokenizer,data,column_name="PretrainSequence"):
    """
    Add ICD codes to the tokenizer
    :param model: model to add the tokens to
    :param tokenizer: tokenizer to add the tokens to
    :param data: the data with the ICD codes
    :return: the updated tokenizer and the model
    """
    ## Add icd9 codes to the tokenizer
    all_codes = list(data[column_name])

    all_codes = [item.split(" ") for item in all_codes]

    all_codes = [item for sublist in all_codes for item in sublist]

    all_codes = list(set(all_codes))

    # check if the tokens are already in the vocabulary
    new_tokens = set(all_codes) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(all_codes)


    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer) + 75)
    return model,tokenizer