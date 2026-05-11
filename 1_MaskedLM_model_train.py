import argparse
import os

import pandas as pd
from transformers import AdamW, get_cosine_schedule_with_warmup, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

from utils.data_handler import load_data
from utils.model_handler import load_model, load_tokenizer, train_tokenizer, add_codes_tokenizer


def custom_data_collator(features,tokenizer=None):
    """
    Custom collator that:
    - Tokenizes the 'HardPrompt' field from each example.
    - Creates labels that are -100 everywhere except at the [MASK] token position,
      where the label is set to the target token (converted to its token ID).
    """
    # Assume global 'tokenizer' and 'args.max_len' are available
    print(features)

    inputs = tokenizer(
        [f["HardPrompts"] for f in features],
        padding=True,
        truncation=True,
        max_length=args.max_len,
        return_tensors="pt"
    )

    # Create labels as a copy of inputs['input_ids']
    labels = inputs["input_ids"].clone()
    # Set all labels to -100 by default (to ignore them in the loss)
    labels[:] = -100

    # Get the token id for [MASK]
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # For each example, find the [MASK] token index and set its label to the target token id.
    for i, feature in enumerate(features):
        input_ids = inputs["input_ids"][i]
        # Find all positions with the mask token; we assume exactly one mask per example.
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) != 1:
            raise ValueError(
                f"Expected exactly one [MASK] token in the input, found {len(mask_indices)} for example {i}.")
        # Convert target token (from our dataset) to its token id.
        target_token = feature["target"]
        target_token_id = tokenizer.convert_tokens_to_ids(target_token)
        # Set the label at the mask position to the target token id.
        labels[i, mask_indices[0]] = target_token_id

    inputs["labels"] = labels
    return inputs

def train(args, model, tokenizer, data_train, data_val, log_dir, model_save, next_word_training=False, mlm_prob=0.25):
    """
    Train a model
    :param args: arguments for training the model in an argparse format
    :param model: model to train
    :param tokenizer: tokenizer to use
    :param data_train: preprocessing training data
    :param data_val: preprocessing validation data
    :param log_dir: directory to save the logs
    :param model_save: directory to save the model
    :param next_word_training: wether to train the model on the next word prediction task
    :param mlm_prob: masked language model probability in case of masked language model training


    """
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if next_word_training:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        if mlm_prob>0:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
        else:
            data_collator = custom_data_collator(tokenizer=tokenizer)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.epochs * len(
                                                    data_train) / args.batch_size)

    training_args = TrainingArguments(
        output_dir=model_save,
        logging_dir=log_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=1

    )

    print(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=data_train,
        eval_dataset=data_val,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()


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
    parser.add_argument("--csv_sep", type=str, default="\t")
    parser.add_argument("--pretrain_sequence_column", type=str, default="PretrainSequence")

    # Model arguments
    parser.add_argument('--model_id', type=str, default="UFNLP/gatortron-base",
                        help="Pretrained model name")
    parser.add_argument('--model_type', type=str, default="bert",
                        help="Pretrained model type")
    parser.add_argument('--model_path', type=str, default="model_id",
                        help="Directory containing the model to load")
    parser.add_argument('--model_out', type=str, default="output",
                        help="Directory containing the model to load")
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help="Directory containing the tokenizer to load")

    # Experiments save path arguments
    parser.add_argument('--output_dir', type=str, default="output",
                        help="Directory to save the predictions")
    parser.add_argument("--experiment_name", type=str, default="gatortron_further_pretrain",
                        help="Name of the experiment")

    # Training arguments
    parser.add_argument('--max_len', type=int, default=512,
                        help="Maximum length of the sequence")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for")
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help="Set the number of warmup steps")
    parser.add_argument('--grad_accum', type=int, default=4,
                        help="Set the number of gradient accumulation steps")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay if we apply some")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Max gradient norm")

    # Arguments for model pretraining
    parser.add_argument('--from_scratch', action='store_true',
                        help='If indicated the model will be trained from scratch. If a tokenizer is not provided'
                             'a tokenizer for the task will also be trained')
    parser.add_argument('--next_word_training', action='store_true',
                        help='If indicated the model will be pretrained on '
                             'the next word prediction task')
    parser.add_argument('--mlm_prob', type=float, default=0.15,
                        help='The percentage of masked tokens. Standard BERT MLM is 0.15. If data already contains [MASK] '
                             'tokens, set this to 0')

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

        # Load data
    data = pd.read_csv(
        args.training_data_dir,
        sep=args.csv_sep)
    data_val = pd.read_csv(
        args.eval_data_dir,
        sep=args.csv_sep)

    if args.from_scratch:
        _, model, config = load_model(model_path, from_scratch=True)
        tokenizer_save = os.path.join(output_dir, "tokenizer")
        if not os.path.exists(tokenizer_save):
            os.makedirs(tokenizer_save)
        if args.tokenizer_path is not None:

            tokenizer = load_tokenizer(args.tokenizer_path)
        else:
            tokenizer_save_path = os.path.join(tokenizer_save, "tokenizer.json")
            tokenizer = train_tokenizer(data, save_path=tokenizer_save_path)
    else:
        tokenizer, model, config = load_model(model_path, from_scratch=False)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    model, tokenizer = add_codes_tokenizer(model, tokenizer, data, column_name=args.pretrain_sequence_column)

    tokenized_train, tokenized_test = load_data(data, data_val, tokenizer)

    train(args, model, tokenizer, tokenized_train, tokenized_test, log_dir, model_save,
          next_word_training=args.next_word_training, mlm_prob=args.mlm_prob)
