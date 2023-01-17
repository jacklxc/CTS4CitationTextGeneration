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

def sampling_scheduler(progress):
    return 1 - progress # Linear

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


def postprocess_text(preds, labels):
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def evaluation(model, batch, gen_kwargs):
    with torch.no_grad():
        batch = {k:torch.tensor(v).to(device) for k,v in batch.items()}
        generated_tokens, retrieved_doc_ids = accelerator.unwrap_model(model).generate(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            allowed_doc_ids = batch["allowed_doc_ids"],
            oracle_doc_ids = batch["oracle_doc_ids"],
            use_oracle = True if args.use_oracle else False,
            **gen_kwargs,
        )

        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.generator.pad_token_id
        )
        labels = batch["labels"]
        #if not args.pad_to_max_length:
        # If we did not pad to max length, we need to pad the labels too
        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.generator.pad_token_id)

        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        labels = accelerator.gather(labels).cpu().numpy()

        #if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        return decoded_preds, decoded_labels, retrieved_doc_ids.tolist()


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
    argparser.add_argument('--passages_path', type=str,
                        default="cited_text_embeddings_citation_mark/cited_papers",
                    )         
    argparser.add_argument('--train_dataset', type=str)#, default="/home/data/XiangciLi/CORWA/annotated_train")
    argparser.add_argument('--distant_dataset', type=str)#, default="/home/data/XiangciLi/CORWA/CORWA_distant")
    argparser.add_argument('--dev_dataset', type=str, default="/home/data/XiangciLi/CORWA/annotated_test_Nov15")

    argparser.add_argument('--train_oracle_ids', type=str, default="sorted_ROUGE_train.jsonl")
    argparser.add_argument('--distant_oracle_ids', type=str, default="sorted_ROUGE_distant.jsonl")
    argparser.add_argument('--dev_oracle_ids', type=str, default="sorted_ROUGE_test.jsonl")
    # /home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl
    # /home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl
    # "/home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl"
    argparser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")
    argparser.add_argument('--num_train_epochs', type=int, default=3,
                           help="Training epoch")
    argparser.add_argument('--max_input_length', type=int,
                           default=512)  # 1024
    argparser.add_argument('--max_combined_length', type=int,
                        default=800)  # 1024
                           
    argparser.add_argument('--max_output_length', type=int, default=256)
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
        default=4,
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
        default="CTS_RAG_span_generation.jsonl",
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
    argparser.add_argument('--freeze_question_encoder', action='store_true')
    argparser.add_argument('--use_oracle', action='store_true')
    argparser.add_argument('--exclude_context', action='store_true')
    argparser.add_argument('--scheduled_sampling', action='store_true')
    argparser.add_argument('--include_conclusion', action='store_true')
    argparser.add_argument('--auto_regressive', action='store_true')
    argparser.add_argument('--dominant_only', action='store_true')
    argparser.add_argument('--reference_only', action='store_true')
    argparser.add_argument('--include_intro', action='store_true')
    argparser.add_argument('--use_fid', action='store_true')
    argparser.add_argument('--cited_abstract_only', action='store_true')
    argparser.add_argument('--prediction_only', action='store_true')
    argparser.add_argument('--allow_reference_CTS', action='store_true')
    
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

    # Creating the Training and Validation dataset for further creation of Dataloader
    
    #if args.distant_dataset is not None:
        #cited_paragraph_ids_path = "cited_text_embeddings/cited_paragraph_ids.jsonl"
        #passages_path = os.path.join("cited_text_embeddings_citation_mark", "cited_papers") 
        #index_path = os.path.join("cited_text_embeddings", "cited_papers_hnsw_index.faiss")
    #else: 
        #cited_paragraph_ids_path = "cited_text_embeddings/selected_cited_paragraph_ids.jsonl"
        #passages_path = os.path.join("cited_text_embeddings_citation_mark", "selected_cited_papers")
        #index_path = os.path.join("cited_text_embeddings", "selected_cited_papers_hnsw_index.faiss")

    #with open(cited_paragraph_ids_path) as f:
    #    cited_paragraph_ids_list = json.load(f)

    #candidate_paragraph_indices = {}
    #for i, pid in enumerate(cited_paragraph_ids_list["ids"]):
    #    link, _ = pid.split("_")
    #    this_link = candidate_paragraph_indices.get(link, [])
    #    this_link.append(i)
    #    candidate_paragraph_indices[link] = this_link  

    cited_dataset = load_from_disk(args.passages_path)
    #cited_dataset.load_faiss_index("embeddings", index_path) 
     
    if args.train_dataset is not None:
        training_dataset = CitationTextGenerationRAGDataset(
            args.train_dataset, tokenizer.question_encoder,
            MAX_SENT_LEN=args.max_input_length,
            include_conclusion=args.include_conclusion,
            include_intro = args.include_intro,
            skip_no_citations = True,
            auto_regressive=args.auto_regressive, 
            include_context = not args.exclude_context,
            dominant_only=args.dominant_only,
            reference_only=args.reference_only,
            related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
            cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
            citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
            limit = args.data_example_limit,
            cited_paper_ids = set(cited_dataset["paper_id"]),
            context_window_size = args.context_window_size,
            rouge_oracle_retrieved_paragraphs = args.train_oracle_ids,
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
             include_context = not args.exclude_context,
             dominant_only=args.dominant_only,
             reference_only=args.reference_only,
             limit = args.data_example_limit,
             cited_paper_ids = set(cited_dataset["paper_id"]),
            context_window_size = args.context_window_size,
            rouge_oracle_retrieved_paragraphs = args.distant_oracle_ids,
            n_docs = args.n_docs,
        )
        #with open("distant_set_CTS_rouge_all.pkl","rb") as f:
        #    distant_dataset = pickle.load(f)
        print("Loaded distant dataset!")
        if args.train_dataset is not None:
            training_dataset.merge(distant_dataset)
        else:
            training_dataset = distant_dataset

    training_set = prepare_dataset(training_dataset, batch_size=args.per_device_train_batch_size)
    
    validation_dataset = CitationTextGenerationRAGDataset(
         args.dev_dataset, tokenizer.question_encoder,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = True,
         auto_regressive=args.auto_regressive, 
         include_context = not args.exclude_context,
         dominant_only=args.dominant_only,
         reference_only=args.reference_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
         cited_paper_ids = set(cited_dataset["paper_id"]),
         context_window_size = args.context_window_size,
         rouge_oracle_retrieved_paragraphs = args.dev_oracle_ids,
         n_docs = args.n_docs,
    )
    validation_set = prepare_dataset(validation_dataset, batch_size=args.per_device_eval_batch_size)

    # load model + enable gradient checkpointing & disable cache for checkpointing
    if args.use_fid:
        args.rag_path = "facebook/rag-sequence-base"

    retriever = RagRetriever.from_pretrained(args.rag_path, index_name="custom", indexed_dataset=cited_dataset)
    retriever.config.max_combined_length = args.max_combined_length
    retriever.generator_tokenizer = gen_tokenizer
    retriever.question_encoder_tokenizer = question_encoder_tokenizer

    oracle_retriever = OracleRetriever(cited_dataset, question_encoder_tokenizer, gen_tokenizer, args.max_combined_length)

    model = CitationSpanGeneratorRAG(
        args.rag_path, 
        args.question_encoder_path, 
        args.generator_path, 
        retriever,
        oracle_retriever = oracle_retriever,
        n_docs=args.n_docs, 
        question_encoder_tokenizer_len = len(tokenizer.question_encoder), 
        generator_tokenizer_len = len(tokenizer.generator),
        use_fid = args.use_fid,
        cited_abstract_only = args.cited_abstract_only,
        reference_abstract_only = not args.allow_reference_CTS,
    )
    # Prepare everything with our `accelerator`.
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))

    model = accelerator.prepare(model)
    
    if args.freeze_question_encoder:
        for param in model.model.question_encoder.parameters():
            param.requires_grad = False

    # Optimizer
    if hasattr(model,"module"):
        optimizer = AdamW(model.module.model.parameters(), lr=args.learning_rate)
    else:
        optimizer = AdamW(model.model.parameters(), lr=args.learning_rate)
    # Prepare everything with our `accelerator`.
    optimizer = accelerator.prepare(optimizer)
    
    train_dataloader = DataLoader(training_set, shuffle=True, collate_fn = data_collator, batch_size=args.per_device_train_batch_size)
    train_dataloader = accelerator.prepare(train_dataloader)
    
    eval_dataloader = DataLoader(validation_set, shuffle=False, collate_fn = data_collator, batch_size=args.per_device_eval_batch_size)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = accelerator.prepare(get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    ))
    
    
    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    tokenizer = accelerator.prepare(tokenizer)
        
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(training_set)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    training_losses = []
    results = []
    for epoch in range(starting_epoch, args.num_train_epochs):
        if not args.prediction_only:
            model.train()
            if args.with_tracking:
                total_loss = 0

            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed stepf
                #torch.cuda.empty_cache() ## Greatly reduces GPU memory usage!
                if args.use_oracle:
                    use_oracle = True
                elif args.scheduled_sampling:
                    cumulative_step = epoch * len(train_dataloader) + step
                    percentage_progress = cumulative_step / (args.num_train_epochs * len(train_dataloader))
                    p_oracle = sampling_scheduler(percentage_progress)
                    #print(p_oracle)
                    use_oracle = random.random() < p_oracle
                else:
                    use_oracle = False
                batch = {k:torch.tensor(v).to(device) if v is not None else None for k,v in batch.items()}
                outputs = model(**batch, use_oracle = use_oracle)

                if type(outputs.loss) == dict:
                    loss = outputs.loss["loss"]
                else:
                    loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.sum().detach().float()
                training_losses.append(loss.sum().detach().float().cpu().numpy())
                progress_bar.set_description(f'Training loss {np.round(np.mean(training_losses[-10:]),4)}')
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss.sum())

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                #if isinstance(checkpointing_steps, int):
                #    if completed_steps % checkpointing_steps == 0:
                #        output_dir = f"step_{completed_steps }"
                #        if args.output_dir is not None:
                #            output_dir = os.path.join(args.output_dir, output_dir)
                #        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

        
        if not args.prediction_only and args.checkpoint_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model).to("cpu")
            #unwrapped_model.model.save_pretrained(args.checkpoint_dir)
            torch.save(unwrapped_model.state_dict(), args.checkpoint_dir)
            #if accelerator.is_main_process:
            #    tokenizer.save_pretrained(args.checkpoint_dir)
            model = model.to(device)
        
        torch.cuda.empty_cache() ## Greatly reduces GPU memory usage!
        model.eval()
        metric = evaluate.load("rouge")
        if args.max_output_length is None:
            args.max_output_length = args.max_output_length

        gen_kwargs = {
            "max_length": args.max_output_length,
            "num_beams": args.num_beams,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty, 
        }

        predictions = []
        gold_labels = []
        retrieved_doc_ids = []

        for step, batch in tqdm(enumerate(eval_dataloader)):
            prediction, gold_label, doc_ids = evaluation(model, batch, gen_kwargs)
            predictions.extend(prediction)
            gold_labels.extend(gold_label)
            retrieved_doc_ids.extend(doc_ids)
            
            metric.add_batch(
                predictions=prediction,
                references=gold_label,
            )
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        logger.info(result)
        

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            results.append(result)
            with open(args.output_dir,"w", encoding='utf8') as f:
                for prediction, gold_label, doc_ids, data in zip(predictions, gold_labels, retrieved_doc_ids, validation_dataset):
                    this_data = copy.deepcopy(data)
                    this_data["prediction"] = prediction
                    this_data["gold_label"] = gold_label
                    this_data["retrieved_doc_ids"] = doc_ids
                    json.dump(this_data,f, ensure_ascii=False)
                    f.write("\n")
        
    with jsonlines.open("RAG.log", mode='a') as writer:
        params = vars(args)
        params["generation_result"] = results
        writer.write(params)
        

