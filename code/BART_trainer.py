import argparse
import os
import pickle
import json

import numpy as np
import torch
from datasets import Dataset as HuggingfaceDataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from dataset import (
    SimpleShortCrossDocumentLMdataset
)

# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def process_data_to_model_inputs(batch):
    inputs = tokenizer(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=True
    )
    outputs = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=args.max_output_length,
        add_special_tokens=True
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch

def get_dict(original_dataset):
    dataset = {}
    keys = original_dataset[0].keys()
    for k in keys:
        dataset[k] = []
    for sample in original_dataset:
        for k in keys:
            dataset[k].append(sample[k])
    return dataset

def prepare_dataset(original_dataset, batch_size=None):
    print("Converting dataset!",flush=True)
    dataset = HuggingfaceDataset.from_dict(get_dict(original_dataset))
    #dataset = HuggingfaceDataset.from_dict(original_dataset.get_dict())
    print("Starting mapping!",flush=True)
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size or args.batch_size,
        remove_columns=["id", "source", "target"],
    )
    # print("End mapping!",flush=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return dataset

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str,
                           default="facebook/bart-base",
                           help="Word embedding file")
    argparser.add_argument('--model_type', type=str,
                       default="facebook/bart-base",
                       help="Word embedding file")
    argparser.add_argument('--dataset', type=str, default="/data/XiangciLi/20200705v1/cs/related_works_year.jsonl")
    argparser.add_argument('--cited_metadata', type=str, default='/data/XiangciLi/20200705v1/cs/cited_metadata.jsonl')
    #argparser.add_argument('--train_dataset', type=str, default="cdlm_train.json")
    #argparser.add_argument('--dev_dataset', type=str, default='cdlm_dev.json')
    # /home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl
    # /home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl
    # "/home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl"
    # argparser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    argparser.add_argument('--epoch', type=int, default=1,
                           help="Training epoch")
    argparser.add_argument('--max_input_length', type=int,
                           default=1024)  # 1024
    argparser.add_argument('--max_output_length', type=int, default=1024)
    argparser.add_argument('--checkpoint', type=str, default="cdlm_bart_base/")
    argparser.add_argument('--batch_size', type=int, default=6) # Per device!
    argparser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")

    args = argparser.parse_args()
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    special_tokens = ['<doc>', '</doc>', '[BOS]']
    additional_special_tokens = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)

    training_set = SimpleShortCrossDocumentLMdataset(args.dataset, 
                                                     tokenizer, train=True, # Train / dev split by year
                                                     MAX_SENT_LEN=args.max_input_length,
                                                     bod_token="<doc>", eod_token="</doc>",
                                                     cited_metadata_path=args.cited_metadata,
                                                     )
    
    # Creating the Training and Validation dataset for further creation of Dataloader
    #print("Processing training set!")
    #with open(args.train_dataset) as f:
    #    training_set = json.load(f)
    training_set = prepare_dataset(training_set)

    print("Loading dev set!")
    validation_set = SimpleShortCrossDocumentLMdataset(args.dataset, 
                                                     tokenizer, train=False,
                                                     MAX_SENT_LEN=args.max_input_length,
                                                     bod_token="<doc>", eod_token="</doc>",
                                                     cited_metadata_path=args.cited_metadata,
                                                     )
    #print("Processing dev set!")
    #with open(args.dev_dataset) as f:
    #    validation_set = json.load(f)
    validation_set = prepare_dataset(validation_set)

    print("Finished dataset!")
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        fp16=True,
        fp16_backend="auto",
        output_dir=args.checkpoint,
        eval_steps=50000,
        logging_steps=250,
        save_steps=500,
        warmup_steps=100,
        save_total_limit=1,
        gradient_accumulation_steps=4,
        prediction_loss_only=True,
        overwrite_output_dir=True,  ###
    )

    # load model + enable gradient checkpointing & disable cache for checkpointing
    model = AutoModelForSeq2SeqLM.from_pretrained(args.repfile,
                                                #gradient_checkpointing=False,
                                                use_cache=False)

    # set generate hyperparameters
    model.config.num_beams = 4
    model.config.max_length = args.max_output_length
    model.config.min_length = 1
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    model.resize_token_embeddings(len(tokenizer))

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=validation_set,
    )

    trainer.train()