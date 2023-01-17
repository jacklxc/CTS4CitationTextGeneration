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

from transformers import (
    AutoTokenizer
)

train = False
bod_token = '<doc>'
eod_token = '</doc>'
max_sent_len = 1024
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
special_tokens = ['<doc>', '</doc>', '[BOS]', '[Dominant]', '[Reference]',
                    '[B_Dominant]', '[E_Dominant]', '[B_Reference]',
                    '[E_Reference]']
additional_special_tokens = {'additional_special_tokens': special_tokens}
tokenizer.add_special_tokens(additional_special_tokens)

related_work_jsons = read_related_work_jsons('/data/XiangciLi/20200705v1/cs/related_works_year.jsonl')
cited_metadata_jsons = read_related_work_jsons('/data/XiangciLi/20200705v1/cs/cited_metadata.jsonl')

n_process = 32
print("# Processes:", n_process)

distributed_related_work_jsons = [{} for i in range(n_process)]
for i, (ID, related_work) in enumerate(related_work_jsons.items()):
    which_p = i % n_process
    distributed_related_work_jsons[which_p][ID] = related_work
            
def sec2hour(span_sec):
    secs = int(span_sec % 60)
    span_min = int(span_sec // 60)
    mins = span_min % 60
    hours = int(span_min // 60)
    return str(hours)+":"+str(mins)+":"+str(secs)

def create_dataset(idx, related_work_jsons):
    local_samples = []
    for ID, related_work in tqdm(related_work_jsons.items()):
        year = related_work["year"]
        if year is None:
            year = 0
        if (train and year <= 2017) or (not train and year == 2018):
            bib_entries = related_work["bib_entries"]
            for i, paragraph in enumerate(related_work["related_work"]):
                inputs = []
                noisy_text, target = makeMLMsample(paragraph["text"], mask_token = tokenizer.mask_token)
                inputs.extend([bod_token, noisy_text, eod_token])
                for citation in paragraph["cite_spans"]:
                    if citation["ref_id"] in bib_entries:
                        this_inputs = deepcopy(inputs)
                        reference_link = bib_entries[citation["ref_id"]][
                            "link"]
                        if reference_link in cited_metadata_jsons:
                            cited_metadata = cited_metadata_jsons[
                                reference_link]
                            title = cited_metadata["title"]
                            if title is None:
                                title = ""
                            abstract = cited_metadata["abstract"]
                            if abstract is None:
                                abstract = ""
                            this_inputs.extend(
                                [bod_token, title, tokenizer.sep_token,
                                 abstract, eod_token])
                            if len(tokenizer(" ".join(this_inputs))[
                                       "input_ids"]) > max_sent_len:
                                continue
                            source = " ".join(this_inputs)
                            local_samples.append({
                                "id": ID + "_" + str(i) + "_" + str(reference_link),
                                "source": source,
                                "target": target
                            })

    with open("cdlm_dataset/"+str(idx)+".jsonl","w") as f:
        for sample in local_samples:
            json.dump(sample, f)
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