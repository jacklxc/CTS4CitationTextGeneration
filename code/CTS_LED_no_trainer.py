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
import jsonlines
from pathlib import Path
from copy import deepcopy

import json
import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric, Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
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
    CitationTextGenerationRetrievedCTSDataset
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
        "--train_dataset", type=str, default="/home/data/XiangciLi/CORWA/annotated_train", help="A csv or a json file containing the training data."
    )
    
    parser.add_argument(
        "--distant_dataset", type=str, default=None, help="A csv or a json file containing the training data." # "/home/data/XiangciLi/CORWA/CORWA_distant"
    )
    
    parser.add_argument(
        "--dev_dataset", type=str, default="/home/data/XiangciLi/CORWA/annotated_test_Nov15", help="A csv or a json file containing the validation data."
    )

    parser.add_argument('--train_retrieved_ids', type=str, default="retrieved_train_doc_ids.jsonl")
    parser.add_argument('--distant_retrieved_ids', type=str, default="retrieved_distant_doc_ids.jsonl")
    parser.add_argument('--dev_retrieved_ids', type=str, default="retrieved_test_doc_ids.jsonl")

    parser.add_argument('--train_oracle_ids', type=str, default="sorted_ROUGE_train.jsonl")
    parser.add_argument('--distant_oracle_ids', type=str, default="sorted_ROUGE_distant.jsonl")
    parser.add_argument('--dev_oracle_ids', type=str, default="sorted_ROUGE_test.jsonl")

    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
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
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    
    parser.add_argument(
        "--dominant_only", action="store_true"
    )
    
    parser.add_argument(
        "--disjoint_support", action="store_true"
    )
    
    parser.add_argument(
        "--include_intro", action="store_true"
    )

    parser.add_argument(
        "--include_conclusion", action="store_true"
    )
    
    parser.add_argument(
        "--skip_no_citations", action="store_true"
    )

    parser.add_argument(
        "--auto_regressive", action="store_true"
    )
    
    parser.add_argument(
        "--exclude_context", action="store_true"
    )
    
    parser.add_argument(
        "--exclude_cited_paper", action="store_true"
    )

    parser.add_argument(
        "--use_oracle", action="store_true"
    )

    parser.add_argument(
        "--scheduled_sampling", action="store_true"
    )

    parser.add_argument(
        "--prediction_only", action="store_true"
    )
    parser.add_argument('--add_keywords', action='store_true')
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
        default=1024,
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
        "--data_example_limit",
        type=int,
        default=None
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
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--n_docs", type=int, default=5
    )

    parser.add_argument(
        "--context_window_size",
        type=int,
        default=2,
    )

    parser.add_argument('--passages_path', type=str, default="cited_text_embeddings_citation_mark/cited_papers")
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="CTS_LED_span_generator.jsonl")
    parser.add_argument("--checkpoint", type=str, default=None, help="Where to store the final model.")
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

def sampling_scheduler(progress):
    return 1 - progress # Linear

class SimpleDataLoader():
    def __init__(self, dataset, shuffle=False):
        self.shuffle = shuffle
        self.dataset = dataset
        
    def __wrap_batch(self, data):
        return {k:v.unsqueeze(0).to(device) for k,v in data.items()}
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        indices = [i for i in range(len(self.dataset))]
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            current_data = self.dataset[idx]
            yield self.__wrap_batch(current_data)

class ScheduledSamplingDataLoader():
    def __init__(self, dataset, oracle_dataset, shuffle=False):
        self.shuffle = shuffle
        self.dataset = dataset
        self.oracle_dataset = oracle_dataset
        self.p_oracle = 1
        
    def __wrap_batch(self, data):
        return {k:v.unsqueeze(0).to(device) for k,v in data.items()}
    
    def __len__(self):
        return min([len(self.dataset), len(self.oracle_dataset)])

    def set_p_oracle(self, p):
        self.p_oracle = p
    
    def __iter__(self):
        indices = [i for i in range(len(self))]
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            if random.random() < self.p_oracle:
                current_data = self.oracle_dataset[idx]
            else:
                current_data = self.dataset[idx]
            yield self.__wrap_batch(current_data)

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
    
    def evaluate(learner, evaluation_data, metric, citation_mark):
        
        def remove_citation_mark(citation_texts, citation_marks):
            cleaned_citation_texts = []
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
            
            #citation_marks = set(citation_mark.split("#"))
            #cleaned_decoded_preds = remove_citation_mark(decoded_preds, citation_marks)
            #cleaned_decoded_labels = remove_citation_mark(decoded_labels, citation_marks)
            
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
        model = LEDForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = LEDForConditionalGeneration.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        
    cited_dataset = load_from_disk(args.passages_path)
    
    validation_set = CitationTextGenerationRetrievedCTSDataset(
         args.dev_dataset, 
         args.dev_retrieved_ids,
         cited_dataset,
         tokenizer,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = args.skip_no_citations,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         #cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
         context_window_size = args.context_window_size,
         n_docs = args.n_docs,
         add_keywords = args.add_keywords,
    )

    validation_oracle_set = CitationTextGenerationRetrievedCTSDataset(
         args.dev_dataset, 
         args.dev_oracle_ids,
         cited_dataset,
         tokenizer,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = args.skip_no_citations,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         #cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
         context_window_size = args.context_window_size,
         n_docs = args.n_docs,
         add_keywords = args.add_keywords,
    )
    
    training_set = CitationTextGenerationRetrievedCTSDataset(
         args.train_dataset, 
         args.train_retrieved_ids,
         cited_dataset,
         tokenizer,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = args.skip_no_citations,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         #cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
         context_window_size = args.context_window_size,
         n_docs = args.n_docs,
         add_keywords = args.add_keywords,
    )

    training_oracle_set = CitationTextGenerationRetrievedCTSDataset(
         args.train_dataset, 
         args.train_oracle_ids,
         cited_dataset,
         tokenizer,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = args.skip_no_citations,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         #cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
         context_window_size = args.context_window_size,
         n_docs = args.n_docs,
         add_keywords = args.add_keywords,
    )
    
    if args.distant_dataset is not None:
        distant_dataset = CitationTextGenerationRetrievedCTSDataset(
             args.distant_dataset, 
             args.distant_retrieved_ids,
             cited_dataset,
             tokenizer,
             MAX_SENT_LEN=args.max_input_length,
             include_conclusion=args.include_conclusion,
             include_intro = args.include_intro,
             skip_no_citations = args.skip_no_citations,
             auto_regressive=args.auto_regressive, 
             dominant_only=args.dominant_only,
             limit = args.data_example_limit,
             context_window_size = args.context_window_size,
             n_docs = args.n_docs,
             add_keywords = args.add_keywords,
        )
        training_set.merge(distant_dataset)

        distant_oracle_dataset = CitationTextGenerationRetrievedCTSDataset(
             args.distant_dataset, 
             args.distant_oracle_ids,
             cited_dataset,
             tokenizer,
             MAX_SENT_LEN=args.max_input_length,
             include_conclusion=args.include_conclusion,
             include_intro = args.include_intro,
             skip_no_citations = args.skip_no_citations,
             auto_regressive=args.auto_regressive, 
             dominant_only=args.dominant_only,
             limit = args.data_example_limit,
             context_window_size = args.context_window_size,
             n_docs = args.n_docs,
             add_keywords = args.add_keywords,
        )
        training_oracle_set.merge(distant_oracle_dataset)
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    train_dataset = Dataset.from_dict(training_set.get_dict())
    eval_dataset = Dataset.from_dict(validation_set.get_dict())
    train_oracle_dataset = Dataset.from_dict(training_oracle_set.get_dict())
    eval_oracle_dataset = Dataset.from_dict(validation_oracle_set.get_dict())

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        
        inputs = examples["source"]
        targets = examples["target"]
        #inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding=padding, truncation=True, add_special_tokens=True)

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
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["id","source","target","citations","citation_links"],
            load_from_cache_file=not args.overwrite_cache,
        )
        
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["source", "target"],
            load_from_cache_file=not args.overwrite_cache,
        )
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask",
                     "labels"],
        )
        eval_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask",
                     "labels"],
        )
        train_oracle_dataset = train_oracle_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["id","source","target","citations","citation_links"],
            load_from_cache_file=not args.overwrite_cache,
        )
        
        eval_oracle_dataset = eval_oracle_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["source", "target"],
            load_from_cache_file=not args.overwrite_cache,
        )
        train_oracle_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask",
                     "labels"],
        )
        eval_oracle_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask",
                     "labels"],
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
    
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    #data_collator = DataCollatorForSeq2Seq(
    #    tokenizer,
    #    model=model,
    #    label_pad_token_id=label_pad_token_id,
    #    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    #)
    # Write a custom dataloader!!
    
    #train_dataloader = DataLoader(
    #    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    #)
    #eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    if args.scheduled_sampling:
        train_dataloader = ScheduledSamplingDataLoader(train_dataset, train_oracle_dataset, shuffle=True)
        print(len(train_dataset), len(train_oracle_dataset))
    elif args.use_oracle:
        train_dataloader = SimpleDataLoader(train_oracle_dataset, shuffle=True)
    else:
        train_dataloader = SimpleDataLoader(train_dataset, shuffle=True)

    if args.use_oracle:
        eval_dataloader = SimpleDataLoader(eval_oracle_dataset, shuffle=False)
    else:
        eval_dataloader = SimpleDataLoader(eval_dataset, shuffle=False)
    
    # Prepare everything with our `accelerator`.
    model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # Prepare everything with our `accelerator`.
    optimizer = accelerator.prepare(optimizer)
    
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    train_metric = load_metric("rouge")
    eval_metric = load_metric("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    
    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }
    
    training_losses = []
    validation_losses = []
    
    for epoch in range(args.num_train_epochs):
        if not args.prediction_only:
            model.train()
            for step, data in enumerate(train_dataloader):
                if args.scheduled_sampling:
                    cumulative_step = epoch * len(train_dataloader) + step
                    percentage_progress = cumulative_step / (args.num_train_epochs * len(train_dataloader))
                    p_oracle = sampling_scheduler(percentage_progress)
                    #print(p_oracle)
                    train_dataloader.set_p_oracle(p_oracle)
                outputs = model(**data)
                loss = outputs.loss
                accelerator.backward(loss)
                training_losses.append(loss.item())
                progress_bar.set_description(f'Training loss {round(np.mean(training_losses[-50:]),4)}')
                
                progress_bar.update(1)
                completed_steps += 1
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if completed_steps >= args.max_train_steps:
                    break
        #compute_metric(train_metric)
        
        model.eval()
        predictions = []
        gold_labels = []
        for step, data in tqdm(enumerate(eval_dataloader)):
            citation_mark = validation_oracle_set[step]["citations"] if args.use_oracle else validation_set[step]["citations"]
            prediction, gold_label = evaluate(model, data, eval_metric, citation_mark)
            predictions.append(prediction)
            gold_labels.append(gold_label)

        logger.info(f'Validation loss {round(np.mean(validation_losses),4)}')
        result = compute_metric(eval_metric)
        logger.info(result)
        
        with open(args.output_dir,"w") as f:
            output_dataset = validation_oracle_set if args.use_oracle else validation_set
            for data, prediction, gold_label in zip(output_dataset, predictions, gold_labels):
                data["prediction"] = prediction
                data["gold_label"] = gold_label
                json.dump(data,f)
                f.write("\n")
                
        if not args.prediction_only and args.checkpoint is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.checkpoint)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.checkpoint)
        
    with jsonlines.open("CTS_LED_span_generation.log", mode='a') as writer:
        params = vars(args)
        params["result"] = result
        del params["lr_scheduler_type"]
        writer.write(params)
        

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()