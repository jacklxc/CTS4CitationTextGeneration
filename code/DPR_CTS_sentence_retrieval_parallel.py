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

from transformers import AutoTokenizer
from retrieval_rag import RagRetriever

n_docs = 40

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

passages_path = os.path.join("cited_text_embeddings_sentence_better", "cited_papers") 
cited_dataset = load_from_disk(passages_path)

special_tokens = ['<doc>', '</doc>', '[BOS]']
additional_special_tokens = {'additional_special_tokens': special_tokens}
question_encoder_tokenizer = AutoTokenizer.from_pretrained("allenai/aspire-sentence-embedder")
question_encoder_tokenizer.add_special_tokens(additional_special_tokens)
gen_tokenizer = AutoTokenizer.from_pretrained("allenai/aspire-sentence-embedder")
gen_tokenizer.add_special_tokens(additional_special_tokens)

retriever = RagRetriever.from_pretrained("facebook/rag-sequence-base", index_name="custom", indexed_dataset=cited_dataset)
retriever.config.max_combined_length = 350
retriever.generator_tokenizer = gen_tokenizer
retriever.question_encoder_tokenizer = question_encoder_tokenizer

base_path = "distant_retrieved_sentence_CTS/"
embedding_file = "distant_query_embeddings_sentence.jsonl"
dataset = []
with open(embedding_file) as f:
    for line in f:
        dataset.append(json.loads(line))

n_process = 16
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

def type2sign(t):
    if t == "R":
        return -1
    else:
        return 1

def retrieve(idx, related_work_jsons):
    with open(base_path + str(idx) + ".jsonl","w") as f:
        for ID, data in tqdm(related_work_jsons.items()):
            allowed_doc_ids = [int(link[1:]) * type2sign(link[0]) for link in data["citation_links"].split("@")]
            input_ids = np.array(question_encoder_tokenizer([data["source"]])["input_ids"])
            docs_dict = retriever(input_ids, np.array(data["pooler_output"]).reshape(1, -1), [allowed_doc_ids], return_tensors="pt", n_docs=n_docs, cited_abstract_only = False)
            data["cited_indices"] = docs_dict["doc_ids"].tolist()[0]
            json.dump(data, f)
            f.write("\n")

def worker(i, related_work_jsons):
    start = time.time()
    print("Starting worker",i)

    retrieve(i, related_work_jsons)

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