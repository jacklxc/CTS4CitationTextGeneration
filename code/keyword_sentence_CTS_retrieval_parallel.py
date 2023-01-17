import multiprocessing 
import sys
import time
import pickle
import re
import json
import collections
import random
import os
from copy import deepcopy
from tqdm import tqdm
from util import *

from datasets import load_from_disk
from rouge_score import rouge_scorer
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import CitationTextGenerationRAGDataset

device = "cpu"
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

passages_path = os.path.join("cited_text_embeddings_sentence_better", "cited_papers") 
cited_dataset = load_from_disk(passages_path)

special_tokens = ['<doc>', '</doc>', '[BOS]']
additional_special_tokens = {'additional_special_tokens': special_tokens}
question_encoder_tokenizer = AutoTokenizer.from_pretrained("allenai/aspire-sentence-embedder")
question_encoder_tokenizer.add_special_tokens(additional_special_tokens)

dataset = CitationTextGenerationRAGDataset(
     "/home/data/XiangciLi/CORWA/CORWA_distant", question_encoder_tokenizer,
     MAX_SENT_LEN=512,
     include_conclusion=False,
     include_intro = False,
     skip_no_citations = True,
     auto_regressive=True, 
     dominant_only=True,
     #related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
     #cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl",
     #citing_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_pdf_parses.jsonl",
     limit = None,
     cited_paper_ids = set(cited_dataset["paper_id"]),
     context_window_size = 2,
)

n_process = 8
print("# Processes:", n_process)

distributed_related_work_jsons = [{} for i in range(n_process)]
for i, candidate in enumerate(dataset):
    which_p = i % n_process
    ID = candidate["id"]
    distributed_related_work_jsons[which_p][ID] = candidate
            
def sec2hour(span_sec):
    secs = int(span_sec % 60)
    span_min = int(span_sec // 60)
    mins = span_min % 60
    hours = int(span_min // 60)
    return str(hours)+":"+str(mins)+":"+str(secs)

def create_dataset(idx, related_work_jsons):
    def extract_keywords(text):
        with torch.no_grad():
            encoded = tokenizer(["summarize: " + text])
            encoded = {k: torch.tensor(v).to(device) for k,v in encoded.items()}
            output_idx = extractor.generate(**encoded)
            outputs = tokenizer.batch_decode(output_idx, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        keywords = outputs[0].split(" ;")
        cleaned_keywords = []
        for keyword in keywords:
            if keyword:
                cleaned_keywords.append(keyword.strip())
        return cleaned_keywords

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    extractor = AutoModelForSeq2SeqLM.from_pretrained("t5_keyword_extractor").to(device)

    ranked_cited_paragraph = {}
    for ID, candidate in tqdm(related_work_jsons.items()):
        keywords = extract_keywords(candidate["target"])
        if len(keywords) == 0:
            keywords = ["Abstract:"]
        query = " @ @ ".join(keywords)
        retrieval_scores = []
        cited_indices = []
        for type_cited_id in candidate["citation_links"].split("@"):
            cited_id = type_cited_id[1:]
            this_paper_bool = np.array(cited_dataset["paper_id"]) == cited_id
            this_paper_indices = np.arange(this_paper_bool.size)[this_paper_bool]
            cited_indices.extend(this_paper_indices.tolist())
            this_paper_paragraphs = cited_dataset[this_paper_indices]
            for paragraph in this_paper_paragraphs["text"]:
                scores = scorer.score(query, paragraph)
                retrival_score = round((scores["rouge1"].recall + scores["rouge2"].recall) / 2, 4)
                retrieval_scores.append(retrival_score)
        cited_indices = np.array(cited_indices)
        sorted_local_indices = np.argsort(retrieval_scores)[::-1]
        sorted_rouge_scores = np.array(retrieval_scores)[sorted_local_indices]
        sorted_cited_paragraph_indices = cited_indices[sorted_local_indices]
        ranked_cited_paragraph[candidate["id"]] = {
            "id": candidate["id"],
            "keywords": keywords,
            "scores": sorted_rouge_scores.tolist(),
            "cited_indices": sorted_cited_paragraph_indices.tolist(),
        }

    with open("keyword_CTS/keyword_sentence_CTS_distant_" + str(idx) + ".jsonl","w") as f:
        for k,v in ranked_cited_paragraph.items():
            json.dump(v,f)
            f.write("\n")

def worker(i, related_work_jsons):
    start = time.time()
    print("Starting worker",i)

    create_dataset(i, related_work_jsons)

    print("Worker",i,"finished in", sec2hour(time.time()-start))
    

if __name__ == "__main__": 
    processes = []
    print("Start splitting!")
    for i, related_work_jsons in zip(range(n_process), distributed_related_work_jsons):
        # creating processes 
        p = multiprocessing.Process(target=worker, args=(i, related_work_jsons)) 
        processes.append(p)
        
    for p in processes:
        # starting process p
        p.start() 

    for p in processes:
        # wait until process p is finished 
        p.join() 

    #final_dataset = created_datasets[0]
    #for data in created_datasets[1:]:
    #    final_dataset = final_dataset.merge(data)
        
    #with open("train_set_CTS_rouge.pkl","wb") as f:
    #    pickle.dump(final_dataset, f)

    print("Done!") 