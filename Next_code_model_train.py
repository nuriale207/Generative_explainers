import argparse
import os

from transformers import AdamW, get_cosine_schedule_with_warmup, TrainingArguments, Trainer

from utils import metrics
from utils.data_handler import DataHandler
from utils.model_handler import load_tokenizer, load_model_multi_label
import pandas as pd
import torch
def save_predictions(output_dir, id2label,predictions,evaluation):
    """
    Save the predictions and evaluation metrics
    :param output_dir: directory to save the predictions
    :param id2label: dictionary to convert the label ids to labels
    :param predictions: predictions from the model
    :param evaluation: evaluation metrics
    :return:
    """
    labels_names = list(id2label.values())
    pred_df = pd.DataFrame(columns=labels_names)
    pred_sigmoid_df = pd.DataFrame(columns=labels_names)
    for prediction in predictions.predictions:
        sigmoid = torch.nn.Sigmoid()
        prediction_bin = sigmoid(torch.Tensor(prediction))
        prediction_bin[prediction_bin > 0.5] = 1
        prediction_bin[prediction_bin <= 0.5] = 0
        prediction_bin = prediction_bin.tolist()
        pred_dict = dict(zip(labels_names, prediction_bin))
        # print(pred_dict)
        # print(pred_df.columns)
        pred_df = pd.concat([pred_df, pd.DataFrame([pred_dict])], ignore_index=True)
        # append the sigmoid predictions
        prediction_sig = sigmoid(torch.Tensor(prediction))
        # turn to list
        prediction_sig = prediction_sig.tolist()
        pred_sig_dict = dict(zip(labels_names, prediction_sig))
        pred_sigmoid_df =  pd.concat([pred_sigmoid_df, pd.DataFrame([pred_sig_dict])], ignore_index=True)

    predictions_file = os.path.join(output_dir, "predictions.csv")

    # Save predictions
    pred_df.to_csv(predictions_file, index=False)

    predictions_sigmoid_file = os.path.join(output_dir, "predictions_raw.csv")
    pred_sigmoid_df.to_csv(predictions_sigmoid_file, index=False)

    evaluation_file = os.path.join(output_dir, "evaluation.txt")

    # Save evaluation
    with open(evaluation_file, "w") as f:
        for key, value in evaluation.items():
            f.write(str(key) + " " + str(value) + "\n")


def train(args,model, tokenizer, data_train,data_val,log_dir,model_save,mode="train",data_test=None):

    """
    Train a model for multi-label classification
    :param args: arguments for training the model in an argparse format
    :param model: model to train
    :param tokenizer: tokenizer to use
    :param data_train: preprocessing training data
    :param data_val: preprocessing validation data
    :param log_dir: log directory
    :param model_save: where to save the model
    :param mode: mode to run the function it can be train, test or both. When mode is test, the function returns the evaluation and predictions
                When mode is both, the function returns the evaluation and predictions for the test data after training the model. When mode is train,
                the function just trains the model
    :param data_test: When mode is test or both, the test data to evaluate the model
    :return:
        evaluation: evaluation metrics
        predictions: model predictions
    """

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.epochs * len(
                                                    data_train) / args.batch_size)

    training_args = TrainingArguments(
        output_dir=model_save,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        learning_rate=args.lr,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=log_dir,  # directory for storing logs
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        fp16=True,
        metric_for_best_model=args.metric_for_best_model,
        adam_epsilon=args.adam_epsilon
    )

    # print(model)
    print("Training model")

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=data_train,  # training dataset
        eval_dataset=data_val,  # evaluation dataset
        compute_metrics=metrics.compute_metrics,  # define metrics function
        optimizers=(optimizer, scheduler)

    )
    if mode=="train":
        trainer.train()
        # Una vez finalizado, guardas el modelo
        final_model_path = os.path.join(model_save, "final_model")
        trainer.save_model(final_model_path)  # ¡Guardas el modelo final aquí!

    elif mode=="test":
        evaluation = trainer.evaluate(data_val)
        predictions = trainer.predict(data_val)

        return evaluation,predictions
    elif mode=="both":
        trainer.train()
        evaluation = trainer.evaluate(data_test)
        predictions = trainer.predict(data_test)
        return evaluation,predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training and prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    parser.add_argument('--training_data_dir', type=str, default="train_data",
                        help="Directory containing the training dataset")
    parser.add_argument('--test_data_dir', type=str, default="test_data",
                        help="Directory containing the test dataset")
    parser.add_argument('--eval_data_dir', type=str, default="eval_data",
                        help="Directory containing the evaluation dataset")
    parser.add_argument('--patients_data_dir', type=str, default="patients_data",
                        help="Directory containing the patients data")
    parser.add_argument("--labels_default_start_index", type=int, default=1,
                        help="Default start index for the labels. If None, the labels will be indexed from 0 to n_labels-1")
    parser.add_argument("--sequence_column",type=str, default="inputsequence",
                        help="The column of the text to train the model, default column is inputsequence")
    parser.add_argument("--csv_sep", type=str, default="\t")


    #Model arguments
    parser.add_argument('--mode', type=str, default="train",
                                help="train or test")
    parser.add_argument('--model_id', type=str, default="allenai/longformer-base-4096",
                        help="Pretrained model name")
    parser.add_argument('--model_type', type=str, default="longformer",
                        help="Pretrained model type")
    parser.add_argument('--model_path', type=str, default="model_id",
                        help="Directory containing the model to load")
    parser.add_argument('--model_out', type=str, default="output",
                        help="Directory containing the model to load")
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help="Directory containing the tokenizer to load")
    parser.add_argument('--metric_for_best_model', type=str, default="recall_micro",
                        help="Directory containing the model to load")

    #Experiments save path arguments
    parser.add_argument('--output_dir', type=str, default="output",
                        help="Directory to save the predictions")
    parser.add_argument("--experiment_name", type=str, default="longformer",
                        help="Name of the experiment")

    #Training arguments
    parser.add_argument('--max_len', type=int, default=512,
                        help="Maximum length of the sequence")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for")
    parser.add_argument('--warmup_steps', type=int, default=125,
                        help="Set the number of warmup steps")
    parser.add_argument('--grad_accum', type=int, default=1,
                        help="Set the number of gradient accumulation steps")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.07,
                        help="Weight decay if we apply some")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Max gradient norm")

    args = parser.parse_args()

    if args.model_path == "model_id":
        model_path = args.model_id
    else:
        model_path = args.model_path

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_save = os.path.join(output_dir, "model")
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    tokenizer_save = os.path.join(output_dir, "tokenizer")
    if not os.path.exists(tokenizer_save):
        os.makedirs(tokenizer_save)

    if args.tokenizer_path is not None:
        # _, model, config = load_model(model_path, code2index_test, index2code_test)
        tokenizer = load_tokenizer(args.tokenizer_path)

    else:
        tokenizer = load_tokenizer(args.model_id,tokenizer_id=args.model_id)
    if args.model_id=="UFNLP/gatortron-base":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if args.mode == "train":
        # Load data
        data_train = DataHandler(args.training_data_dir,args.batch_size,None,tokenizer,max_length=args.max_len,
                                  labels_default_start_index=args.labels_default_start_index,
                                 sequence_column=args.sequence_column,data_sep=args.csv_sep)


        train_data=data_train.get_tokenized_data()
        label2id,id2label=data_train.get_labels_dict()

        data_val = DataHandler(args.eval_data_dir,args.batch_size,None,tokenizer,max_length=args.max_len,
                                  labels_default_start_index=args.labels_default_start_index,data_sep=args.csv_sep,
                               sequence_column=args.sequence_column)
        val_data=data_val.get_tokenized_data()

        data_test=DataHandler(args.test_data_dir,args.batch_size,None,tokenizer,max_length=args.max_len,
                                  labels_default_start_index=args.labels_default_start_index,
                                 sequence_column=args.sequence_column,data_sep=args.csv_sep)
        test_data=data_test.get_tokenized_data()


        _,model,config=load_model_multi_label(model_path, label2id, id2label,load_tokenizer=False)


        train(args, model, tokenizer, train_data, val_data,log_dir,model_save)

    elif args.mode == "test":


        data_test = DataHandler(args.test_data_dir, args.batch_size, None, tokenizer, max_length=args.max_len,
                                 labels_default_start_index=args.labels_default_start_index,data_sep=args.csv_sep,
                                sequence_column=args.sequence_column)

        test_data = data_test.get_tokenized_data()
        label2id,id2label=data_test.get_labels_dict()

        # Create output directory inside the experiment directory
        output_dir = os.path.join(args.output_dir, args.experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create test directory inside the output directory
        test_dir = os.path.join(output_dir, "test")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        _,model,config=load_model_multi_label(model_path, label2id, id2label,load_tokenizer=False)

        evaluation, predictions=train(args, model, tokenizer, test_data, test_data, log_dir, model_save,mode="test")

        save_predictions(test_dir,id2label,predictions,evaluation)

    elif args.mode == "both":
        # Load data
        data_train = DataHandler(args.training_data_dir, args.batch_size, None, tokenizer, max_length=args.max_len,
                                 labels_default_start_index=args.labels_default_start_index,
                                 sequence_column=args.sequence_column, data_sep=args.csv_sep)

        train_data = data_train.get_tokenized_data()
        label2id, id2label = data_train.get_labels_dict()

        data_val = DataHandler(args.eval_data_dir, args.batch_size, None, tokenizer, max_length=args.max_len,
                               labels_default_start_index=args.labels_default_start_index,
                               sequence_column=args.sequence_column, data_sep=args.csv_sep)
        val_data = data_val.get_tokenized_data()

        data_test = DataHandler(args.test_data_dir, args.batch_size, None, tokenizer, max_length=args.max_len,
                                labels_default_start_index=args.labels_default_start_index,
                                sequence_column=args.sequence_column, data_sep=args.csv_sep)
        test_data = data_test.get_tokenized_data()

        _, model, config = load_model_multi_label(model_path, label2id, id2label,load_tokenizer=False)
        # Create output directory inside the experiment directory
        output_dir = os.path.join(args.output_dir, args.experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create test directory inside the output directory
        test_dir = os.path.join(output_dir, "test")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        evaluation, predictions=train(args, model, tokenizer, train_data, val_data, log_dir, model_save,mode="both")
        save_predictions(test_dir,id2label,predictions,evaluation)
