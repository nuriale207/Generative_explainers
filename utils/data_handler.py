import pandas as pd
import numpy as np


from datasets import Dataset

def preprocess_data(examples,tokenizer):
    """
    Preprocess the data using the tokenizer for the model
    :param examples: an object of type Dataset containing the data it must contain the Icd_codes column
    :param tokenizer: the tokenizer to use
    :return: the encoded data
    """
    # take a batch of texts
    text = examples["PretrainSequence"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    # add labels
    # labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
    # # create numpy array of shape (batch_size, num_labels)
    # labels_matrix = np.zeros((len(text), len(self.labels)))
    # # fill numpy array
    # for idx, label in enumerate(self.labels):
    #     labels_matrix[:, idx] = labels_batch[label]

    # encoding["labels"] = labels_matrix.tolist()

    return encoding

def load_data(data,data_val,tokenizer):
    """
    Load the data and preprocess it using the tokenizer
    :param data: a pandas dataframe containing the data
    :param data_val: a pandas dataframe containing the validation data
    :param tokenizer: the tokenizer to use
    :return:
        tokenized_train: the tokenized training data
        tokenized_val: the tokenized validation data
    """
    data_filter = data.filter(["PretrainSequence"])
    data_test_filter = data_val.filter(["PretrainSequence"])

    train_dataset = Dataset.from_pandas(data_filter)
    test_dataset = Dataset.from_pandas(data_test_filter)

    # print(train_dataset)

    tokenized_train = train_dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)
    tokenized_train.set_format("torch")

    tokenized_val = test_dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)
    tokenized_val.set_format("torch")


    return tokenized_train,tokenized_val



class DataHandler:
    """
    Class to handle the data
    """
    def __init__(self, data_path, batch_size, labels_ids,tokenizer,max_length, sequence_column="inputsequence",labels_default_start_index=None,data_sep="\t"):
        """
        Initialize the DataHandler
        :param data_path: path to the data csv file
        :param batch_size: batch size
        :param labels_ids: list of labels ids
        :param tokenizer: tokenizer to use
        :param max_length: maximum length of the sequence
        :param labels_default_start_index: dataframes columns index where the labels start
        :param data_sep: data separator of the csv file
        """
        print("DataHandler init")
        self.data_path = data_path
        self.batch_size = batch_size
        self.labels_ids = str(labels_ids)
        self.data_sep=data_sep
        self.sequence_column=sequence_column.lower()
        # self.sequence_column.lower()
        if not labels_ids is None:
            self.labels_ids = labels_ids.replace("\n", "")
            self.labels_ids = labels_ids.rstrip("\n")
            self.labels_ids=list(labels_ids.split(',')) #convert to list
            self.labels_ids= [label.lower() for label in self.labels_ids]
            print("LABELS IDS: "+str(self.labels_ids))
        elif not labels_default_start_index is None:
            print(self.data_path)
            print(self.data_sep)
            df_train = pd.read_csv(self.data_path, index_col = False, sep=self.data_sep)
            # df_train=pd.read_csv("/tartalo01/users/nlebena001/gscratch3/Datasets/MIMIC_IV/full_mullenbach/sectioned/"
            #                      "discharge_notes_raw_sectioned_reduced_icds_5508_labels_train.csv", sep="\t")


            # print("LABELS DEFAULT START INDEX: " + str(labels_default_start_index))
            # print("Dataframe columns: "+str(df_train.columns))
            # print("DataSep: "+str(self.data_sep))
            self.labels_ids = list(df_train.columns[labels_default_start_index:])
            self.labels_ids= [label.lower() for label in self.labels_ids]
        self.tokenizer= tokenizer
        self.max_length = max_length
        self.get_loader()
    #funciona?

    def get_loader(self):
        """
        Get the data loader. This function must be called after the initialization of the class to get the data and
        labels from the csv file
        :return: nothing
        """
        df_train = pd.read_csv(self.data_path, sep=self.data_sep)
        #Get the data and labels columns
        # print("Labels ids list ")
        # print(["text"]+self.labels_ids)
        # print("Dataframe columns: ")
        # print(df_train.columns)

        df_train.columns = [x.lower() for x in df_train.columns]

        df_train = df_train.filter([self.sequence_column]+self.labels_ids, axis=1)

        #Create a dataset from the pandas dataframe
        train_data = Dataset.from_pandas(df_train)
        #Save data in local variable
        self.data = train_data
        #Get the labels
        labels = [label for label in train_data.features.keys() if label not in [self.sequence_column]]
        self.labels=labels
        # print("Labels: ")
        # print(self.labels)

    def preprocess_data(self,examples):
        """
        Preprocess the data using the tokenizer
        :param examples: dataframes examples containing inputsequence and labels
        :return:
        """
        # take a batch of texts
        text = examples[self.sequence_column]
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

    def get_tokenized_data(self):
        """
        Get the tokenized data
        :return: tokenized data
        """
        tokenized_train = self.data.map(self.preprocess_data, batched=True)
        tokenized_train.set_format("torch")
        return tokenized_train

    def get_labels_dict(self):
        """
        Get the id2label and label2id dictionaries
        :return: id2label and label2id dictionaries
        """
        id2label = {idx:label for idx, label in enumerate(self.labels)}
        label2id = {label:idx for idx, label in enumerate(self.labels)}
        return label2id,id2label