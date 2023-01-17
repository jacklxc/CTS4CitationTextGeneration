#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path
from copy import deepcopy
import jsonlines
import json
import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
#from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    #DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
    LEDForConditionalGeneration,
)
from transformers.utils.versions import require_version

from dataset import (
    CitationTextGenerationSciSummDataset
)

logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")

    parser.add_argument(
        "--validation_file", type=str, help="A csv or a json file containing the validation data."
    )
   
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=8192,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="allenai/led-base-16384",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="allenai/led-base-16384",
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    parser.add_argument(
        "--auto_regressive",
        action="store_true",
    )
    
    parser.add_argument(
        "--annotated_CTS",
        action="store_true",
    )
    parser.add_argument(
        "--agreed_only",
        action="store_true",
    )
    
    parser.add_argument(
        "--output_file", type=str,
        default="CTS_predictions.jsonl",
    )

    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=2,
    )
    
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
    )

    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    args = parser.parse_args()

    return args

def compute_metric(metric):
    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}
    return result

def data_collator(features):
    label_pad_token_id = -100
    N_tokens = [feature["input_ids"].shape[0] for feature in features]
    max_input_len = max(N_tokens)
    max_target_len = max([feature["labels"].shape[0] for feature in features])
    ids = []
    citations = []
    labels = []
    all_input_ids = []
    attention_masks = []
    global_attention_masks = []
    for feature in features:
        #ids.append(feature["id"].unsqueeze(0))
        #citations.append(feature["citations"].unsqueeze(0))
        label = torch.ones((1,max_target_len)).long() * label_pad_token_id
        label[0,:feature["labels"].shape[0]] = feature["labels"]
        labels.append(label)
        input_ids = torch.zeros((1,max_input_len)).long()
        input_ids[0,:feature["input_ids"].shape[0]] = feature["input_ids"]
        all_input_ids.append(input_ids)
        attention_mask = torch.zeros((1,max_input_len)).long()
        attention_mask[0,:feature["attention_mask"].shape[0]] = feature["attention_mask"]
        attention_masks.append(attention_mask)

        global_attention_mask = torch.zeros((1,max_input_len)).long()
        global_attention_mask[0,:feature["global_attention_mask"].shape[0]] = feature["global_attention_mask"]
        global_attention_masks.append(global_attention_mask)
        
    return {
        #"id": torch.cat(ids),
        #"citations": torch.cat(citations),
        "labels": torch.cat(labels),
        "input_ids": torch.cat(all_input_ids),
        "attention_mask": torch.cat(attention_masks),
        "global_attention_mask": torch.cat(global_attention_masks)
    }


def main():
    
    def predict(learner, evaluation_data):
        generated_tokens = accelerator.unwrap_model(learner).generate(
            evaluation_data["input_ids"],
            attention_mask=evaluation_data["attention_mask"],
            **gen_kwargs,
        )

        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )

        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded_preds
    
    def evaluate(learner, evaluation_data, metric, citation_marks=None):
        
        def remove_citation_mark(citation_texts, citation_marks):
            cleaned_citation_texts = []
            fragments = []
            for mark in citation_marks:
                fragments.extend(mark.split())
            citation_marks += fragments
            for citation_text in citation_texts:
                for mark in citation_marks:
                    citation_text = citation_text.replace(mark, "")
                cleaned_citation_texts.append(citation_text)
            return cleaned_citation_texts
        
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(learner).generate(
                evaluation_data["input_ids"],
                attention_mask=evaluation_data["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = evaluation_data["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(evaluation_data["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            if citation_marks:
                citation_marks = set(citation_mark.split("#"))
                cleaned_decoded_preds = remove_citation_mark(decoded_preds, citation_marks)
                cleaned_decoded_labels = remove_citation_mark(decoded_labels, citation_marks)
                metric.add_batch(predictions=cleaned_decoded_preds, references=cleaned_decoded_labels)
            else:
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            
            return decoded_preds, decoded_labels
    
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        #datasets.utils.logging.set_verbosity_warning()
        #transformers.utils.logging.set_verbosity_info()
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
        
    special_tokens = ['<doc>', '</doc>', '[BOS]', '[Dominant]', '[Reference]',
                      '[B_Dominant]', '[E_Dominant]', '[B_Reference]',
                      '[E_Reference]']
    additional_special_tokens = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        
    
    validation_set = CitationTextGenerationSciSummDataset(
        args.validation_file, tokenizer,
        auto_regressive=args.auto_regressive, 
        context_window_size = args.context_window_size,
        annotated_CTS = args.annotated_CTS,
        n_docs = args.n_docs,
        agreed_only = args.agreed_only,
    )
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    eval_dataset = Dataset.from_dict(validation_set.get_dict())

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        
        inputs = examples["source"]
        targets = examples["target"]
        #inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True, add_special_tokens=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # since above lists are references, the following line changes the 0 index for all samples
        if "led-" in args.tokenizer_name:
         
            # create 0 global_attention_mask lists
            model_inputs["global_attention_mask"] = [
                [0 for _ in range(len(model_inputs["input_ids"][i_batch]))] for i_batch in range(len(model_inputs["input_ids"]))
            ]
            
            special_token_ids = set(tokenizer.additional_special_tokens_ids)
            special_token_ids.add(tokenizer.cls_token_id)
            special_token_ids.add(tokenizer.sep_token_id)
            for i_batch in range(len(model_inputs["input_ids"])):
                for i_token in range(len(model_inputs["input_ids"][i_batch])):
                    if model_inputs["input_ids"][i_batch][i_token] in special_token_ids:
                        model_inputs["global_attention_mask"][i_batch][i_token] = 1

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["source", "target"],
            load_from_cache_file=not args.overwrite_cache,
        )

        eval_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask",
                     "labels"]#,"citations","id"],
        )
    # Log a few random samples from the training set:
    #for index in random.sample(range(len(train_dataset)), 1):
    #    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    #label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    #data_collator = DataCollatorForSeq2Seq(
    #    tokenizer,
    #    model=model,
    #    label_pad_token_id=label_pad_token_id,
    #    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    #)

    eval_dataloader = DataLoader(eval_dataset, collate_fn = data_collator, batch_size=args.per_device_eval_batch_size)
    
    #eval_dataloader = SimpleDataLoader(eval_dataset, validation_set, shuffle=False)
    
    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

  
    eval_metric = load_metric("rouge")

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    
    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }
        
    model.eval()
    predictions = []
    gold_labels = []
    for data in tqdm(eval_dataloader):
        prediction, gold_label = evaluate(model, data, eval_metric)
        predictions.extend(prediction)
        gold_labels.extend(gold_label)

    result = compute_metric(eval_metric)
    logger.info(result)
    
    with open(args.output_file,"w") as f:
        for data, prediction, gold_label in zip(validation_set, predictions, gold_labels):
            data["prediction"] = prediction
            data["gold_label"] = gold_label
            json.dump(data,f)
            f.write("\n")
        
    #with jsonlines.open("predictions.log", mode='a') as writer:
    #    params = vars(args)
    #    params["result"] = result
    #    writer.write(params)
        

if __name__ == "__main__":
    main()