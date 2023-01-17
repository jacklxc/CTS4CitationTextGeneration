import argparse
import os
import pickle

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
    SingleAbstractCitationTextGenerationDataset
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


def process_data_to_model_inputs(batch,
                                 special_tokens=['[Dominant]', '[Reference]']):
    # tokenize the inputs and labels

    additional_special_tokens_lookup = {token: idx for token, idx in
                                        zip(tokenizer.additional_special_tokens,
                                            tokenizer.additional_special_tokens_ids)}
    special_token_ids = set(
        [additional_special_tokens_lookup[token] for token in special_tokens])
    special_token_ids.add(tokenizer.mask_token_id)

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

def prepare_dataset(original_dataset, batch_size=None):
    # print("Converting dataset!",flush=True)
    dataset = HuggingfaceDataset.from_dict(original_dataset.get_dict())
    # print("Starting mapping!",flush=True)
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size or args.batch_size,
        remove_columns=["id", "source", "target"],
    )
    # print("End mapping!",flush=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask",
                 "labels"],
    )
    return dataset


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str,
                           default="bart_base_span_generator_cdlm/checkpoint-83000",
                           help="Word embedding file")
    argparser.add_argument('--train_dataset', type=str, default="/home/data/XiangciLi/CORWA/annotated_train")
    argparser.add_argument('--distant_dataset', type=str, default="/home/data/XiangciLi/CORWA/CORWA_distant")
    argparser.add_argument('--dev_dataset', type=str, default="/home/data/XiangciLi/CORWA/annotated_test_Nov15")
    # /home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl
    # /home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl
    # "/home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl"
    # argparser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    argparser.add_argument('--epoch', type=int, default=3,
                           help="Training epoch")
    argparser.add_argument('--max_input_length', type=int,
                           default=1024)
    argparser.add_argument('--max_output_length', type=int, default=1024)
    argparser.add_argument('--checkpoint', type=str, default="bart_base_span_generator_cdlm/")
    argparser.add_argument('--batch_size', type=int, default=6) # Per device!
    argparser.add_argument(
        "--data_example_limit",
        type=int,
        default=None
    )

    argparser.add_argument('--include_conclusion', action='store_true')
    argparser.add_argument('--auto_regressive', action='store_true')
    argparser.add_argument('--dominant_only', action='store_true')
    argparser.add_argument('--include_intro', action='store_true')
    argparser.add_argument('--skip_no_citations', action='store_true')
    
    argparser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")

    args = argparser.parse_args()
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    special_tokens = ['<doc>', '</doc>', '[BOS]', '[Dominant]', '[Reference]',
                      '[B_Dominant]', '[E_Dominant]', '[B_Reference]',
                      '[E_Reference]']
    additional_special_tokens = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)

    # Creating the Training and Validation dataset for further creation of Dataloader

    training_set = SingleAbstractCitationTextGenerationDataset(
         args.train_dataset, tokenizer,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = args.skip_no_citations,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
    )

    if args.distant_dataset is not None:
        distant_dataset = SingleAbstractCitationTextGenerationDataset(
             args.distant_dataset, tokenizer,
             MAX_SENT_LEN=args.max_input_length,
             include_conclusion=args.include_conclusion,
             include_intro = args.include_intro,
             skip_no_citations = args.skip_no_citations,
             auto_regressive=args.auto_regressive, 
             dominant_only=args.dominant_only,
             limit = args.data_example_limit,
        )
        #with open("distant_set_CTS_rouge_all.pkl","rb") as f:
        #    distant_dataset = pickle.load(f)
        print("Loaded distant dataset!")
        training_set.merge(distant_dataset)

    training_set = prepare_dataset(training_set)
    validation_set = SingleAbstractCitationTextGenerationDataset(
         args.dev_dataset, tokenizer,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = args.skip_no_citations,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
    )
    validation_set = prepare_dataset(validation_set)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        fp16=True,
        fp16_backend="auto",
        output_dir=args.checkpoint,
        eval_steps=1000,
        logging_steps=250,
        save_steps=500,
        warmup_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        prediction_loss_only=True,
        overwrite_output_dir=True,  ###
    )

    # load model + enable gradient checkpointing & disable cache for checkpointing
    led = AutoModelForSeq2SeqLM.from_pretrained(args.repfile,
                                                gradient_checkpointing=False,
                                                use_cache=False)

    # set generate hyperparameters
    led.config.num_beams = 4
    led.config.max_length = args.max_output_length
    led.config.min_length = 1
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    led.resize_token_embeddings(len(tokenizer))

    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=validation_set,
    )

    trainer.train()