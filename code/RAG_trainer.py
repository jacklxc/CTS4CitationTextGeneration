import argparse
import os
import pickle
import json

import numpy as np
import torch
from datasets import Dataset as HuggingfaceDataset
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)

import rouge

from dataset import (
    CitationTextGenerationRAGDataset
)

from models import CitationSpanGeneratorRAG

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

    # Assuming batch_size = 1
    batch["allowed_doc_ids"] = []
    for example in batch["citation_links"]:
        citation_links = example.split("@")
        allowed_doc_ids = []
        for link in citation_links:
            if link in candidate_paragraph_indices:
                allowed_doc_ids.extend(candidate_paragraph_indices[link])
    batch["allowed_doc_ids"].append(allowed_doc_ids) 

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

def prepare_dataset(original_dataset, batch_size=None):
    # print("Converting dataset!",flush=True)
    dataset = HuggingfaceDataset.from_dict(original_dataset.get_dict())
    # print("Starting mapping!",flush=True)
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size or args.batch_size,
        remove_columns=["id", "source", "target", "citation_links"],
    )
    # print("End mapping!",flush=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "allowed_doc_ids", #"global_attention_mask",
                "labels"],
    )
    return dataset


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str,
                           default="initial_full_text_citation_span_generator",
                           help="Word embedding file")
    argparser.add_argument('--train_dataset', type=str, default="/home/data/XiangciLi/CORWA/annotated_train")
    argparser.add_argument('--distant_dataset', type=str)#, default="/home/data/XiangciLi/CORWA/CORWA_distant")
    argparser.add_argument('--dev_dataset', type=str, default="/home/data/XiangciLi/CORWA/annotated_test_Nov15")
    # /home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl
    # /home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl
    # "/home/data/XiangciLi/20200705v1/cs/related_works_year.jsonl"
    # argparser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    argparser.add_argument('--epoch', type=int, default=3,
                           help="Training epoch")
    argparser.add_argument('--max_input_length', type=int,
                           default=512)  # 1024
    argparser.add_argument('--max_output_length', type=int, default=1024)
    argparser.add_argument('--checkpoint', type=str, default="citation_span_generator/")
    argparser.add_argument('--batch_size', type=int, default=1) # Per device!
    argparser.add_argument(
        "--data_example_limit",
        type=int,
        default=None
    )

    argparser.add_argument('--include_conclusion', action='store_true')
    argparser.add_argument('--auto_regressive', action='store_true')
    argparser.add_argument('--dominant_only', action='store_true')
    argparser.add_argument('--include_intro', action='store_true')
    #argparser.add_argument('--skip_no_citations', action='store_true')
    
    argparser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")

    args = argparser.parse_args()
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = RagTokenizer.from_pretrained(args.repfile)
    special_tokens = ['<doc>', '</doc>', '[BOS]', '[Dominant]', '[Reference]',
                  '[B_Dominant]', '[E_Dominant]', '[B_Reference]',
                  '[E_Reference]', '[CTS]']
    additional_special_tokens = {'additional_special_tokens': special_tokens}
    tokenizer.question_encoder.add_special_tokens(additional_special_tokens)
    tokenizer.generator.add_special_tokens(additional_special_tokens)

    # Creating the Training and Validation dataset for further creation of Dataloader
    
    if args.distant_dataset is not None:
        cited_paragraph_ids_path = "cited_text_embeddings/cited_paragraph_ids.jsonl"
        passages_path = os.path.join("cited_text_embeddings", "cited_papers")
        index_path = os.path.join("cited_text_embeddings", "cited_papers_hnsw_index.faiss")
    else: 
        cited_paragraph_ids_path = "cited_text_embeddings/selected_cited_paragraph_ids.jsonl"
        passages_path = os.path.join("cited_text_embeddings", "selected_cited_papers")
        index_path = os.path.join("cited_text_embeddings", "selected_cited_papers_hnsw_index.faiss")

    with open(cited_paragraph_ids_path) as f:
        cited_paragraph_ids_list = json.load(f)

    candidate_paragraph_indices = {}
    for i, pid in enumerate(cited_paragraph_ids_list["ids"]):
        link, _ = pid.split("_")
        this_link = candidate_paragraph_indices.get(link, [])
        this_link.append(i)
        candidate_paragraph_indices[link] = this_link   


    training_set = CitationTextGenerationRAGDataset(
         args.train_dataset, tokenizer.question_encoder,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = True,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
         cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
         citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
         limit = args.data_example_limit,
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
             limit = args.data_example_limit,
        )
        #with open("distant_set_CTS_rouge_all.pkl","rb") as f:
        #    distant_dataset = pickle.load(f)
        print("Loaded distant dataset!")
        training_set.merge(distant_dataset)

    training_set = prepare_dataset(training_set)
    
    validation_set = CitationTextGenerationRAGDataset(
         args.dev_dataset, tokenizer.question_encoder,
         MAX_SENT_LEN=args.max_input_length,
         include_conclusion=args.include_conclusion,
         include_intro = args.include_intro,
         skip_no_citations = True,
         auto_regressive=args.auto_regressive, 
         dominant_only=args.dominant_only,
         related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
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

    cited_dataset = load_from_disk(passages_path)
    cited_dataset.load_faiss_index("embeddings", index_path)

    # load model + enable gradient checkpointing & disable cache for checkpointing
    model = CitationSpanGeneratorRAG(args.repfile, cited_dataset, n_docs=100)
    model.model.question_encoder.resize_token_embeddings(len(tokenizer.question_encoder))
    model.model.generator.resize_token_embeddings(len(tokenizer.generator))

    # set generate hyperparameters
    model.model.config.num_beams = 4
    model.model.config.max_length = args.max_output_length
    model.model.config.min_length = 1
    model.model.config.length_penalty = 2.0
    model.model.config.early_stopping = True
    model.model.config.no_repeat_ngram_size = 3

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer.generator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=validation_set,
    )

    trainer.train()