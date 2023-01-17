import logging
import os
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from collections import OrderedDict
from tqdm import tqdm

from nltk import sent_tokenize
from util import *

import torch
from datasets import Features, Sequence, Value, load_dataset, Dataset, load_from_disk

import faiss
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    AutoModel
)

logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ctx_encoder_sentence_better" #"allenai/aspire-sentence-embedder"
tokenizer_name = "allenai/aspire-sentence-embedder"
file_path = "cited_text_embeddings_sentence_better"

def get_citation_mark(metadata):
    if len(metadata["authors"]) > 0:
        first_author = metadata["authors"][0]["last"]
    else:
        first_author = ""
    return first_author + "@" + str(metadata.get("year",""))

def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation='longest_first', max_length=512, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}

def load_custom_DPR_embedding(model_path):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    aspire_model = AutoModel.from_pretrained(model_path)
    renamed_aspire_weights = OrderedDict()
    for k,v in aspire_model.state_dict().items():
        renamed_aspire_weights["ctx_encoder.bert_model."+k] = v
    model = DPRContextEncoder.from_pretrained(model_path)
    model.load_state_dict(renamed_aspire_weights, strict=True)
    logging.getLogger("transformers").setLevel(logging.INFO)
    return model


cited_papers = {}
with open("/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl") as f:
    for line in f:
        paper_dict = json.loads(line)
        cited_papers[paper_dict["paper_id"]] = paper_dict
        
cited_metadata = {}
with open("/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl") as f:
    for line in f:
        paper_dict = json.loads(line)
        cited_metadata[paper_dict["paper_id"]] = paper_dict
        
cited_paper_jsons = []
for key, cited_paper in tqdm(cited_papers.items()):
    citation_mark = get_citation_mark(cited_metadata[key])
    cited_paragraph = {
        "paper_id": cited_paper["paper_id"],
        "paragraph_id": "t0",
        "title": citation_mark,
        "text": "Title: " + cited_paper["title"],
    }
    if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
        cited_paper_jsons.append(cited_paragraph)
    
    for i, paragraph in enumerate(cited_paper["abstract"]):
        segmented_sentences = patch_sent_tokenize(sent_tokenize(paragraph["text"]))
        for si, sent in enumerate(segmented_sentences):
            cited_paragraph = {
                "paper_id": cited_paper["paper_id"],
                "paragraph_id": "a" + str(i) + "#" + str(si),
                "title": citation_mark,
                "text": paragraph["section"] + ": " + sent,
            }
            if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                cited_paper_jsons.append(cited_paragraph)
            
    for i, paragraph in enumerate(cited_paper["body_text"]):
        segmented_sentences = patch_sent_tokenize(sent_tokenize(paragraph["text"]))
        for si, sent in enumerate(segmented_sentences):
            cited_paragraph = {
                "paper_id": cited_paper["paper_id"],
                "paragraph_id": "b" + str(i) + "#" + str(si),
                "title": citation_mark,
                "text": paragraph["section"] + ": " + sent,
            }
            if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                cited_paper_jsons.append(cited_paragraph)

if not os.path.exists(file_path):
    os.mkdir(file_path)
raw_dataset_path = os.path.join(file_path, "cited_paragraphs.jsonl")
            
with open(raw_dataset_path,"w") as f:
    for paper in cited_paper_jsons:
        json.dump(paper,f)
        f.write("\n")
        
dataset = load_dataset(
        "json", data_files=[raw_dataset_path], split="train"
    )

#ctx_encoder = load_custom_DPR_embedding(model_name).to(device=device)
ctx_encoder = DPRContextEncoder.from_pretrained(model_name).to(device=device)
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(tokenizer_name)
new_features = Features(
    {"paper_id": Value("string"), "paragraph_id": Value("string"), "title": Value("string"), "text": Value("string"), "embeddings": Sequence(Value("float32"))}
)  # optional, save as float32 instead of float64 to save space
dataset = dataset.map(
    partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
    batched=True,
    batch_size=128,
    features=new_features,
)

passages_path = os.path.join(file_path, "cited_papers")
dataset.save_to_disk(passages_path)