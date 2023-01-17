import multiprocessing 
import sys
import time
import pickle
import re
import json
import collections
import random
import os

from dataset import (
    CitationTextGenerationDataset
)

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
special_tokens = ['<doc>', '</doc>', '[BOS]', '[Dominant]', '[Reference]',
                    '[B_Dominant]', '[E_Dominant]', '[B_Reference]',
                    '[E_Reference]']
additional_special_tokens = {'additional_special_tokens': special_tokens}
tokenizer.add_special_tokens(additional_special_tokens)

n_processes = 5
print("# Processes:", n_processes)

chunk_size = 11465 // n_processes + 1 
print("Chunk size:", chunk_size)

def sec2hour(span_sec):
    secs = int(span_sec % 60)
    span_min = int(span_sec // 60)
    mins = span_min % 60
    hours = int(span_min // 60)
    return str(hours)+":"+str(mins)+":"+str(secs)

def create_dataset(i):
    start = i * chunk_size
    end = (i+1) * chunk_size
    distant_dataset = CitationTextGenerationDataset(
            "/home/data/XiangciLi/CORWA/CORWA_distant", tokenizer,
            MAX_SENT_LEN=4096,
            include_conclusion=False,
            include_intro = False,
            skip_no_citations = False,
            auto_regressive=False, 
            dominant_only=True,
            start = start,
            limit = end,
            best_CTS_rouge = True,
    )
    
    with open("distant_set_CTS_rouge/"+str(i)+".pkl","wb") as f:
        pickle.dump(distant_dataset, f)

def worker(i):
    start = time.time()
    print("Starting worker",i)

    create_dataset(i)

    print("Worker",i,"finished in", sec2hour(time.time()-start))
    

if __name__ == "__main__": 
    processes = []
    print("Start splitting!")
    for i in range(n_processes):
        # creating processes 
        p = multiprocessing.Process(target=worker, args=(i, )) 
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