from transformers import (
    AutoTokenizer,
)

from dataset import (
    CTS_DPR_Dataset
)

from tqdm import tqdm

import json

model_name = "allenai/aspire-sentence-embedder"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
special_tokens = ['<doc>', '</doc>', '[BOS]']
additional_special_tokens = {'additional_special_tokens': special_tokens}
tokenizer.add_special_tokens(additional_special_tokens)

data_set = CTS_DPR_Dataset(
     "/home/data/XiangciLi/CORWA/CORWA_distant", tokenizer,
     MAX_SENT_LEN=99999,
     include_conclusion=False,
     include_intro=False,
     skip_no_citations=True,
     auto_regressive=True, 
     dominant_only=False,
     #related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
     #cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
     #cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
     #citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
     limit = None,
     start = 0,
     include_context = True,
    context_window_size = 2,
    sentence_level = True,
)

with open("DPR_CORWA_sentence_distant_dataset.jsonl","w") as f:
    for data in data_set:
        json.dump(data,f)
        f.write("\n")

with open("DPR_CORWA_sentence_distant.json","w") as f:
    json.dump(data_set.make_dpr_inputs("CORWA_distant", top_k=40),f)