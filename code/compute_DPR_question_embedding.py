import argparse
import logging
import math
import os
import random
import pickle
from pathlib import Path
import copy
import re
import json
import jsonlines
import datasets
datasets.utils.disable_progress_bar()
import nltk
import numpy as np
import torch
from torch.distributions import Categorical
from datasets import Dataset as HuggingfaceDataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
from sklearn.metrics import f1_score, precision_score, recall_score
import transformers
transformers.logging.set_verbosity_error()
from accelerate import Accelerator, DistributedDataParallelKwargs
#from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
    BartForConditionalGeneration,
    RagTokenizer,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    DPRQuestionEncoder,
)
from transformers.utils.versions import require_version

from dataset import (
    CitationTextGenerationRAGDataset
)

from models import CitationSpanGeneratorRAG, OracleRetriever
from retrieval_rag import RagRetriever

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

link_pad_id = 0

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    #special_tokens=['[Dominant]', '[Reference]']
    #additional_special_tokens_lookup = {token: idx for token, idx in
    #                                    zip(tokenizer.additional_special_tokens,
    #                                        tokenizer.additional_special_tokens_ids)}
    #special_token_ids = set(
    #    [additional_special_tokens_lookup[token] for token in special_tokens])
    #special_token_ids.add(tokenizer.mask_token_id)

    inputs = tokenizer.question_encoder(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=True
    )
    outputs = tokenizer.generator(
        batch["target"],
        padding="longest",
        truncation=True,
        max_length=args.max_output_length,
        add_special_tokens=True
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    def type2sign(t):
        if t == "R":
            return -1
        else:
            return 1
    batch_allowed_doc_ids = []
    for example in batch["citation_links"]:
        citation_links = example.split("@")
        allowed_doc_ids = [int(link[1:]) * type2sign(link[0]) for link in citation_links]
        #for link in citation_links:
        #    if link in candidate_paragraph_indices:
        #        allowed_doc_ids.extend(candidate_paragraph_indices[link])
        batch_allowed_doc_ids.append(allowed_doc_ids) 

    max_n_links = max([len(links) for links in batch_allowed_doc_ids])
    batch["allowed_doc_ids"] = []
    for links in batch_allowed_doc_ids:
        batch["allowed_doc_ids"].append(links + [link_pad_id] * (max_n_links - len(links)))

    batch_retrieved_doc_ids = []
    for example in batch["retrieved_doc_ids"]:
        retrieved_ids = example.split("@")
        retrieved_indices = [int(idx) for idx in retrieved_ids]
        batch_retrieved_doc_ids.append(retrieved_indices) 
    batch["oracle_doc_ids"] = batch_retrieved_doc_ids

    # create 0 global_attention_mask lists
    #batch["global_attention_mask"] = len(batch["input_ids"]) * [
    #    [0 for _ in range(len(batch["input_ids"][0]))]
    #]

    # since above lists are references, the following line changes the 0 index for all samples
    #for i_batch in range(len(batch["input_ids"])):
    #    for i_token in range(len(batch["input_ids"][0])):
    #        if batch["input_ids"][i_batch][i_token] in special_token_ids:
    #            batch["global_attention_mask"][i_batch][i_token] = 1

    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    #batch["labels"] = [
    #    [-100 if token == tokenizer.generator.pad_token_id else token for token in labels]
    #    for labels in batch["labels"]
    #]
    return batch
    
def data_collator(features):
    label_pad_token_id = tokenizer.generator.pad_token_id
    N_tokens = [feature["input_ids"].shape[0] for feature in features]
    max_input_len = max(N_tokens)
    max_target_len = max([feature["labels"].shape[0] for feature in features])
    max_n_links = max([feature["allowed_doc_ids"].shape[0] for feature in features])
    labels = []
    all_input_ids = []
    attention_masks = []
    allowed_doc_ids = []
    oracle_doc_ids = []
    for feature in features:
        links = torch.ones((1,max_n_links)).long() * link_pad_id
        links[0,:feature["allowed_doc_ids"].shape[0]] = feature["allowed_doc_ids"]
        allowed_doc_ids.append(links)
        oracle_ids = torch.ones((1,args.n_docs)).long() * -1
        oracle_ids[0,:feature["oracle_doc_ids"].shape[0]] = feature["oracle_doc_ids"]
        oracle_doc_ids.append(oracle_ids)
        label = torch.ones((1,max_target_len)).long() * label_pad_token_id
        label[0,:feature["labels"].shape[0]] = feature["labels"]
        labels.append(label)
        input_ids = torch.zeros((1,max_input_len)).long()
        input_ids[0,:feature["input_ids"].shape[0]] = feature["input_ids"]
        all_input_ids.append(input_ids)
        attention_mask = torch.zeros((1,max_input_len)).long()
        attention_mask[0,:feature["attention_mask"].shape[0]] = feature["attention_mask"]
        attention_masks.append(attention_mask)

    return {
        "allowed_doc_ids": torch.cat(allowed_doc_ids),
        "labels": torch.cat(labels),
        "input_ids": torch.cat(all_input_ids),
        "attention_mask": torch.cat(attention_masks),
        "oracle_doc_ids": torch.cat(oracle_doc_ids),
    }


def prepare_dataset(original_dataset, batch_size=None):
    with accelerator.main_process_first():
        # print("Converting dataset!",flush=True)
        dataset = HuggingfaceDataset.from_dict(original_dataset.get_dict())
        # print("Starting mapping!",flush=True)
        dataset = dataset.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size or args.batch_size,
            remove_columns=["id", "source", "target", "citation_links", "retrieved_doc_ids"],
        )
        # print("End mapping!",flush=True)
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "allowed_doc_ids", "oracle_doc_ids", #"global_attention_mask",
                    "labels"],
        )
    return dataset


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(
        description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--rag_path', type=str,
                           default="facebook/rag-sequence-base",
                        )
    argparser.add_argument('--generator_path', type=str,
                           default="facebook/bart-base",
                        )
    argparser.add_argument('--generator_tokenizer_path', type=str,
                        default="facebook/bart-base",
                    )
    argparser.add_argument('--question_encoder_path', type=str,
                        default="dpr_question_encoder",
                    )
    argparser.add_argument('--question_encoder_tokenizer_path', type=str,
                        default="allenai/aspire-sentence-embedder",
                    )
    argparser.add_argument('--train_dataset', type=str)#, default="/home/data/XiangciLi/CORWA/annotated_train")
    argparser.add_argument('--distant_dataset', type=str)#, default="/home/data/XiangciLi/CORWA/CORWA_distant")
    argparser.add_argument('--dev_dataset', type=str, default="/home/data/XiangciLi/CORWA/annotated_test_Nov15")

    # /home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl
    # /home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl
    # "/home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl"
    argparser.add_argument('--max_input_length', type=int,
                           default=512)  # 1024
    argparser.add_argument('--max_combined_length', type=int,
                        default=1024)  # 1024
                           
    argparser.add_argument('--max_output_length', type=int, default=128)
    argparser.add_argument('--per_device_train_batch_size', type=int, default=1)
    argparser.add_argument('--per_device_eval_batch_size', type=int, default=1) # Per device!
    argparser.add_argument(
        "--data_example_limit",
        type=int,
        default=None
    )

    argparser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    argparser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    argparser.add_argument(
        "--context_window_size",
        type=int,
        default=2,
    )

    argparser.add_argument(
        "--lr_scheduler_type",
        #type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    argparser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    argparser.add_argument(
        "--n_docs", type=int, default=5
    )

    argparser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    argparser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    argparser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="CTS_RAG_span_generator.model",
        help="If the training should continue from a checkpoint folder.",
    )

    argparser.add_argument(
        "--output_dir",
        type=str,
        default="retrieved_doc_ids.jsonl",
        help="If the training should continue from a checkpoint folder.",
    )
    
    argparser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )

    argparser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    
    argparser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
    )
    
    argparser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
    )
    
    argparser.add_argument('--passages_path', type=str,
                    default="cited_text_embeddings_citation_mark/cited_papers",
                )   

    argparser.add_argument('--freeze_question_encoder', action='store_true')
    argparser.add_argument('--use_oracle', action='store_true')
    argparser.add_argument('--scheduled_sampling', action='store_true')
    argparser.add_argument('--include_conclusion', action='store_true')
    argparser.add_argument('--auto_regressive', action='store_true')
    argparser.add_argument('--dominant_only', action='store_true')
    argparser.add_argument('--reference_only', action='store_true')
    argparser.add_argument('--include_intro', action='store_true')
    argparser.add_argument('--use_fid', action='store_true')
    argparser.add_argument('--cited_abstract_only', action='store_true')
    argparser.add_argument('--prediction_only', action='store_true')
    
    argparser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")

    args = argparser.parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
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

    torch.backends.cudnn.deterministic = True

    tokenizer = RagTokenizer.from_pretrained(args.rag_path)
    special_tokens = ['<doc>', '</doc>', '[BOS]']
    additional_special_tokens = {'additional_special_tokens': special_tokens}

    gen_tokenizer = AutoTokenizer.from_pretrained(args.generator_tokenizer_path)
    gen_tokenizer.add_special_tokens(additional_special_tokens)
    question_encoder_tokenizer = AutoTokenizer.from_pretrained(args.question_encoder_tokenizer_path)
    question_encoder_tokenizer.add_special_tokens(additional_special_tokens)
    tokenizer.question_encoder = question_encoder_tokenizer
    tokenizer.generator = gen_tokenizer

    cited_dataset = load_from_disk(args.passages_path)
     
    if args.train_dataset is not None:
        training_dataset = CitationTextGenerationRAGDataset(
            args.train_dataset, tokenizer.question_encoder,
            MAX_SENT_LEN=args.max_input_length,
            include_conclusion=args.include_conclusion,
            include_intro = args.include_intro,
            skip_no_citations = True,
            auto_regressive=args.auto_regressive, 
            dominant_only=args.dominant_only,
            reference_only=args.reference_only,
            related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
            cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
            citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
            limit = args.data_example_limit,
            cited_paper_ids = set(cited_dataset["paper_id"]),
            context_window_size = args.context_window_size,
            n_docs = args.n_docs,
        )

    if args.distant_dataset is not None:
        distant_dataset = CitationTextGenerationRAGDataset(
             args.distant_dataset, tokenizer.question_encoder,
             MAX_SENT_LEN=args.max_input_length,
             include_conclusion=args.include_conclusion,
             include_intro = args.include_intro,
             skip_no_citations = True,
             auto_regressive=args.auto_regressive, 
             dominant_only=args.dominant_only,
             reference_only=args.reference_only,
             limit = args.data_example_limit,
             cited_paper_ids = set(cited_dataset["paper_id"]),
            context_window_size = args.context_window_size,
            n_docs = args.n_docs,
        )
        #with open("distant_set_CTS_rouge_all.pkl","rb") as f:
        #    distant_dataset = pickle.load(f)
        print("Loaded distant dataset!")
        if args.train_dataset is not None:
            training_dataset.merge(distant_dataset)
        else:
            training_dataset = distant_dataset

    training_set = prepare_dataset(training_dataset, batch_size=1)

    # load model + enable gradient checkpointing & disable cache for checkpointing
    if args.use_fid:
        args.rag_path = "facebook/rag-sequence-base"

    train_dataloader = DataLoader(training_set, shuffle=False, collate_fn = data_collator, batch_size=1)
    train_dataloader = accelerator.prepare(train_dataloader)

    tokenizer = accelerator.prepare(tokenizer)
    progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_local_main_process)
    model = DPRQuestionEncoder.from_pretrained(args.question_encoder_path)
    model = accelerator.prepare(model)
    model.eval()
    retrieved_doc_ids = []
    with open(args.output_dir,"w", encoding='utf8') as f:
        for step, batch in enumerate(train_dataloader):
            this_data = copy.deepcopy(training_dataset[step])
            # We need to skip steps until we reach the resumed stepf
            #torch.cuda.empty_cache() ## Greatly reduces GPU memory usage!
            if args.use_oracle:
                use_oracle = True
            else:
                use_oracle = False
            #batch = {k:torch.tensor(v).to(device) if v is not None else None for k,v in batch.items()}
            question_hidden_states = model(input_ids = batch["input_ids"].to(device), attention_mask = batch["attention_mask"].to(device)).pooler_output
            pooler_output = question_hidden_states["pooler_output"].detach().cpu().numpy().squeeze(0).tolist()
            
            progress_bar.update(1)
            this_data["pooler_output"] = pooler_output
            json.dump(this_data,f, ensure_ascii=False)
            f.write("\n")