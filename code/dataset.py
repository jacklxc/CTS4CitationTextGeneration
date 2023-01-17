from glob import glob
from collections import OrderedDict
from rouge_score import rouge_scorer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from multiprocessing import Pool
from torch.utils.data import Dataset
from tqdm import tqdm
from util import *
import random
import pickle
from copy import deepcopy
import numpy as np
import torch
import scispacy
import spacy

class CitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                 search_CTS = False,
                 best_CTS_rouge = False,
                 search_context = False,
                 ):
        if best_CTS_rouge:
            search_CTS = True

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)
            
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        #self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        self.dataset = []
        self.samples = []
        
        self.lens = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try: 
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        # "offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        if not dominant_only or span["span_type"] == "Dominant":
                            source = []
                            if include_intro:
                                source.append(introduction_text)
                                source.append(tokenizer.sep_token)                                
                            context_before = paragraph[:span["char_start"]].replace(
                                "[BOS] ", "")
                            context_after = paragraph[span["char_end"]:].replace(
                                "[BOS] ", "")
                            target = paragraph[
                                    span["char_start"]:span["char_end"]].replace(
                                "[BOS] ", "")
                            if include_context:
                                source.append(context_before)
                            if len(span["span_citation_mapping"]["Dominant"]) > 0:
                                source.append("[Dominant]")
                            else:
                                source.append("[Reference]")
                            if not auto_regressive:
                                source.append(context_after)
                            
                            self.lens.append(len(tokenizer.tokenize(" ".join(source))))
                            
                            citation_types = {}
                            cited_paragraphs = OrderedDict()
                            citation_marks_flag = False
                            #high_quality_span = True

                            for span_type in ["Dominant", "Reference"]:
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    citation_marks_flag = True
                                    if link and link in self.cited_paper:
                                        cited_paragraphs[link] = {}
                                        citation_types[link] = span_type
                                        abstract, high_quality_abstract = get_title_abstract(self.cited_paper[link],
                                                                include_result=include_conclusion)
                                        cited_paragraphs[link]["a"] = abstract
                            
                            if best_CTS_rouge:
                                filtered_cited_paragraphs = OrderedDict()
                                for candidate_link in cited_paragraphs:
                                    candidate_CTS_scores = []
                                    for para_id, cited_paragraph in cited_paragraphs[candidate_link].items():
                                        scores = scorer.score(target,cited_paragraph)
                                        candidate_CTS_scores.append((para_id, scores["rougeL"].fmeasure))
                                    sorted_candidate_CTS_scores = sorted(candidate_CTS_scores, key=lambda x: x[-1])[::-1]
                                    best_key = sorted_candidate_CTS_scores[0][0]
                                    best_CTS = cited_paragraphs[candidate_link][best_key]
                                    filtered_cited_paragraphs[candidate_link] = {"b": best_CTS}
                                cited_paragraphs = filtered_cited_paragraphs

                            if skip_no_citations and not citation_marks_flag:
                                continue

                            for candidate_link in cited_paragraphs:
                                for para_id, cited_paragraph in cited_paragraphs[candidate_link].items():
                                    citation_marks = []
                                    this_source = deepcopy(source)
                                    for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                                                        ["[B_Dominant]", "[B_Reference]"],
                                                                        ["[E_Dominant]", "[E_Reference]"]):
                                        for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                            this_source.append(b_span)
                                            this_source.append(citation_mark)
                                            this_source.append(tokenizer.sep_token)
                                            citation_marks.append(citation_mark)

                                            if candidate_link == link:
                                                this_source.append(truncate_string(cited_paragraph, tokenizer, self.max_abstract_length)) ###############
                                            elif link and link in cited_paragraphs and "b" in cited_paragraphs[link]:
                                                this_source.append(truncate_string(cited_paragraphs[link]["b"], tokenizer, self.max_abstract_length))
                                            elif link and link in cited_paragraphs and "a" in cited_paragraphs[link]:
                                                this_source.append(truncate_string(cited_paragraphs[link]["a"], tokenizer, self.max_abstract_length))
                                            this_source.append(e_span)

                                            
                                    if search_context:
                                        citing_abstract, _ = get_title_abstract(self.citing_paper[citing_paper_id],
                                                                include_result=False)
                                        citing_para_i = "a"
                                        this_context_source = deepcopy(this_source)
                                        this_context_source = [citing_abstract, tokenizer.sep_token] + this_context_source
                                        this_context_source = " ".join(this_context_source)

                                        if len(tokenizer.tokenize(this_context_source)) <= self.max_sent_len:
                                            span_id = citing_para_i+"#"+paragraph_id + "_" + str(i_span)
                                            for link in cited_paragraphs:
                                                span_id += "@"+citation_types[link][0]+"_"+link + "_"
                                                if link == candidate_link:
                                                    span_id += str(para_id)
                                                elif best_CTS_rouge:
                                                    span_id += "b"
                                                else:
                                                    span_id += "a"

                                            self.samples.append({
                                                "id": span_id,
                                                "source": this_context_source,
                                                "target": target,
                                                "citations": "#".join(citation_marks)
                                            })
                                        
                                        for citing_para_i, paragraph_obj in enumerate(self.citing_paper[citing_paper_id]["body_text"]):
                                            if "related" in paragraph_obj["section"].lower():
                                                continue
                                            this_context_source = deepcopy(this_source)
                                            this_context_source = [paragraph_obj["text"], tokenizer.sep_token] + this_context_source
                                            this_context_source = " ".join(this_context_source)

                                            if len(tokenizer.tokenize(this_context_source)) <= self.max_sent_len:
                                                span_id = str(citing_para_i)+"#"+paragraph_id + "_" + str(i_span)
                                                for link in cited_paragraphs:
                                                    span_id += "@"+citation_types[link][0]+"_"+link + "_"
                                                    if link == candidate_link:
                                                        span_id += str(para_id)
                                                    elif best_CTS_rouge:
                                                        span_id += "b"
                                                    else:
                                                        span_id += "a"

                                                self.samples.append({
                                                    "id": span_id,
                                                    "source": this_context_source,
                                                    "target": target,
                                                    "citations": "#".join(citation_marks)
                                                })
                                    else:
                                        this_source = " ".join(this_source)

                                        if len(tokenizer.tokenize(this_source)) <= self.max_sent_len:
                                            span_id = paragraph_id + "_" + str(i_span)
                                            for link in cited_paragraphs:
                                                span_id += "@"+citation_types[link][0]+"_"+link + "_"
                                                if link == candidate_link:
                                                    span_id += str(para_id)
                                                elif best_CTS_rouge:
                                                    span_id += "b"
                                                else:
                                                    span_id += "a"

                                            self.samples.append({
                                                "id": span_id,
                                                "source": this_source,
                                                "target": target,
                                                "citations": "#".join(citation_marks)
                                            })

            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        #del self.cited_metadata 
        del self.cited_paper
        #del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset
    
class CTSCitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                 best_CTS_lookup = None,
                 best_context_pid_lookup = None,
                 ):

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)
            
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        #self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)
        
        if best_CTS_lookup:
            with open(best_CTS_lookup,"rb") as f:
                best_CTS_lookup = pickle.load(f)
                
        if best_context_pid_lookup:
            with open(best_context_pid_lookup,"rb") as f:
                best_context_pid_lookup = pickle.load(f)
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        self.dataset = []
        self.samples = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try: 
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        # "offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        if not dominant_only or span["span_type"] == "Dominant":
                            source = []
                            if include_intro:
                                source.append(introduction_text)
                                source.append(tokenizer.sep_token)                                
                            context_before = paragraph[:span["char_start"]].replace(
                                "[BOS] ", "")
                            context_after = paragraph[span["char_end"]:].replace(
                                "[BOS] ", "")
                            target = paragraph[
                                    span["char_start"]:span["char_end"]].replace(
                                "[BOS] ", "")
                            if include_context:
                                source.append(context_before)
                            if len(span["span_citation_mapping"]["Dominant"]) > 0:
                                source.append("[Dominant]")
                            else:
                                source.append("[Reference]")
                            if not auto_regressive:
                                source.append(context_after)

                            citation_types = {}
                            cited_paragraphs = OrderedDict()
                            citation_marks_flag = False
                            #high_quality_span = True

                            for span_type in ["Dominant", "Reference"]:
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    citation_marks_flag = True
                                    if link and link in self.cited_paper:
                                        cited_paragraphs[link] = {}
                                        citation_types[link] = span_type

                                        if best_CTS_lookup and span_type[0]+"_"+link in best_CTS_lookup[paragraph_id + "_" + str(i_span)]:
                                            span_id = paragraph_id + "_" + str(i_span)
                                            cited_pid, _ = best_CTS_lookup[span_id][span_type[0]+"_"+link]
                                            if cited_pid == "a":
                                                abstract, _ = get_title_abstract(self.cited_paper[link],
                                                                    include_result=include_conclusion)
                                                cited_paragraphs[link]["a"] = abstract
                                            else:
                                                cited_paragraphs[link][cited_pid] = self.cited_paper[link]["body_text"][int(cited_pid)]["text"]
                                        else:
                                            abstract, high_quality_abstract = get_title_abstract(self.cited_paper[link],
                                                                    include_result=include_conclusion)
                                            cited_paragraphs[link]["a"] = abstract

                            if skip_no_citations and not citation_marks_flag:
                                continue

                            citation_marks = []
                            for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                                                ["[B_Dominant]", "[B_Reference]"],
                                                                ["[E_Dominant]", "[E_Reference]"]):
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    if link in self.cited_paper:
                                        source.append(b_span)
                                        source.append(citation_mark)
                                        source.append(tokenizer.sep_token)
                                        citation_marks.append(citation_mark)
                                        if link in cited_paragraphs:
                                            cited_pid = list(cited_paragraphs[link].keys())[0]
                                            source.append(truncate_string(cited_paragraphs[link][cited_pid], tokenizer, self.max_abstract_length))
                                        else:
                                            cited_abstract, _ = get_title_abstract(self.cited_paper[link],
                                                            include_result=False)
                                            source.append(truncate_string(cited_abstract, tokenizer, self.max_abstract_length))
                                        source.append(e_span)

                            if best_context_pid_lookup and paragraph_id + "_" + str(i_span) in best_context_pid_lookup:
                                span_id = paragraph_id + "_" + str(i_span)
                                context_pid, _ = best_context_pid_lookup[span_id]
                                if context_pid == "a":
                                    best_context, _ = get_title_abstract(self.citing_paper[citing_paper_id],
                                                        include_result=False)
                                else:
                                    best_context = self.citing_paper[citing_paper_id]["body_text"][int(context_pid)]["text"]
                                    source = [truncate_string(best_context, tokenizer, self.max_abstract_length), tokenizer.sep_token] + source

                                span_id = context_pid+"#"+paragraph_id + "_" + str(i_span)
                            else:
                                span_id = paragraph_id + "_" + str(i_span)
                            source = " ".join(source)
                            if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                                for link in cited_paragraphs:
                                    if link in self.cited_paper:
                                        span_id += "@"+citation_types[link][0]+"_"+link + "_"
                                        cited_pid = list(cited_paragraphs[link].keys())[0]
                                        span_id += str(cited_pid)

                                self.samples.append({
                                    "id": span_id,
                                    "source": source,
                                    "target": target,
                                    "citations": "#".join(citation_marks)
                                })


            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        #del self.cited_metadata 
        del self.cited_paper
        del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitedTextSpanSelectionDataset(Dataset):
    def __init__(self, path_name: str, score_path, tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                ):
        CTS_token = "[CTS]"
        CTS_token_id = tokenizer(CTS_token)['input_ids'][1]
        # Deal with pre-computed loss first.

        def read_span_id(span_id):
            cited_id_list = span_id.split("@")[1:]
            for cited_id in cited_id_list:
                _, cited_paper_id, para_id = cited_id.split("_")
                if para_id != "a":
                    return cited_paper_id, para_id
            return cited_paper_id, para_id

        candidate_loss = {}
        with open(score_path) as f:
            for line in f:
                d = json.loads(line)
                candidate_loss[d["id"]] = 2**(-d["loss"])

        span_series = {}
        for k,v in candidate_loss.items():
            span_id = k.split("@")[0]
            scores = span_series.get(span_id,{})
            scores[k] = v
            span_series[span_id] = scores

        normalized_span_series_by_citation = {}
        for k,v in span_series.items():
            example_id = list(v.keys())[0]
            cited_paper_ids = [cited_id.split("_")[1] for cited_id in example_id.split("@")[1:]]
            loss_lookup = {ID: {} for ID in cited_paper_ids}
            for full_span_id, loss in v.items():
                cited_paper_id, para_id = read_span_id(full_span_id)
                if para_id == "a":
                    for cited_paper_id, this_loss_lookup in loss_lookup.items():
                        this_loss_lookup["a"] = loss
                else:
                    loss_lookup[cited_paper_id][para_id] = loss
            normalized_span_series_by_citation[k] = loss_lookup

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        #self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        self.dataset = []
        self.samples = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        self.missing = 0
        self.success = 0

        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try: 
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        # "offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        span_id = paragraph_id + "_" + str(i_span)
                        if not dominant_only or span["span_type"] == "Dominant":
                            if span_id not in normalized_span_series_by_citation: ##################################################################
                                self.missing += 1
                                continue
                            self.success += 1
                            source = []
                            if include_intro:
                                source.append(introduction_text)
                                source.append(tokenizer.sep_token)                                
                            context_before = paragraph[:span["char_start"]].replace(
                                "[BOS] ", "")
                            context_after = paragraph[span["char_end"]:].replace(
                                "[BOS] ", "")
                            target = paragraph[
                                    span["char_start"]:span["char_end"]].replace(
                                "[BOS] ", "")
                            if include_context:
                                source.append(context_before)
                            if len(span["span_citation_mapping"]["Dominant"]) > 0:
                                source.append("[Dominant]") #########################################################
                                #source.append(target)
                                #source.append("[Dominant]")
                            else:
                                source.append("[Reference]")
                                #source.append(target)
                                #source.append("[Reference]")
                            if not auto_regressive:
                                source.append(context_after)

                            citation_types = {}
                            cited_paragraphs = OrderedDict()
                            citation_marks_flag = False
                            #high_quality_span = True

                            for span_type in ["Dominant", "Reference"]:
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    citation_marks_flag = True
                                    if link and link in self.cited_paper:
                                        cited_paragraphs[link] = {}
                                        citation_types[link] = span_type
                                        abstract, high_quality_abstract = get_title_abstract(self.cited_paper[link])
                                        cited_paragraphs[link]["a"] = abstract
                                        for para_i, paragraph_obj in enumerate(self.cited_paper[link]["body_text"]):
                                            cited_paragraphs[link][str(para_i)] = paragraph_obj["section"] + " | " +paragraph_obj["text"]

                            if skip_no_citations and not citation_marks_flag:
                                continue

                            for candidate_link in cited_paragraphs:
                                citation_marks = []
                                this_source = deepcopy(source)
                                visited_link = set([])
                                for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                                                    ["[B_Dominant]", "[B_Reference]"],
                                                                    ["[E_Dominant]", "[E_Reference]"]):
                                    for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                        if link in visited_link:
                                            continue # For rare cases, same cited paper appear in both dominant and reference spans.
                                        visited_link.add(link)
                                        this_source.append(b_span)
                                        this_source.append(citation_mark)
                                        this_source.append(tokenizer.sep_token)
                                        citation_marks.append(citation_mark)

                                        if candidate_link == link:
                                            CTS_target_scores = []
                                            for para_id, cited_paragraph in cited_paragraphs[candidate_link].items():
                                                this_source.append(CTS_token) 
                                                this_source.append(cited_paragraph)
                                                #if span_id in normalized_span_series_by_citation and link in normalized_span_series_by_citation[span_id] and para_id in normalized_span_series_by_citation[span_id][link]:
                                                CTS_target_scores.append(normalized_span_series_by_citation[span_id][link][para_id]) 
                                                #else:
                                                #    CTS_target_scores.append(0)
                                        elif link and link in cited_paragraphs and "a" in cited_paragraphs[link]:
                                            this_source.append(truncate_string(cited_paragraphs[link]["a"], tokenizer, self.max_abstract_length))
                                        this_source.append(e_span)
        
                                this_source = " ".join(this_source)

                                # Normalize target scores
                                maximum = np.max(CTS_target_scores)
                                minimum = np.min(CTS_target_scores)
                                CTS_target_scores = [(s - minimum) / (maximum - minimum + 1e-5) for s in CTS_target_scores]
                                #mean = np.mean(CTS_target_scores)
                                #std = np.std(CTS_target_scores)
                                #CTS_target_scores = [(s - mean) / (std + 1e-6) for s in CTS_target_scores]
                                CTS_target_scores = torch.softmax(torch.tensor(CTS_target_scores),0).tolist()

                                input_ids = tokenizer(this_source)["input_ids"]
                                if len(CTS_target_scores) >1 and len(input_ids) <= self.max_sent_len:
                                    this_span_id = paragraph_id + "_" + str(i_span)
                                    for link in cited_paragraphs:
                                        this_span_id += "@"+citation_types[link][0]+"_"+link + "_"
                                        if link == candidate_link:
                                            this_span_id += "s"
                                        else:
                                            this_span_id += "a"

                                    #if this_span_id == "195767626_1_0_2@R_205032138_s":
                                    #    print(input_ids)
                                    #    print(target)
                                    #    print(CTS_target_scores)

                                    token_scores = []
                                    pointer = 0
                                    for token_id in input_ids:
                                        if token_id == CTS_token_id:
                                            token_scores.append(CTS_target_scores[pointer])
                                            pointer += 1
                                        else:
                                            token_scores.append(0.5) ############

                                    self.samples.append({
                                        "id": this_span_id,
                                        "source": this_source,
                                        "target": target,
                                        "citations": "#".join(citation_marks),
                                        "CTS_target_scores": CTS_target_scores,
                                        "token_target_scores": token_scores,
                                    })

            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        #del self.cited_metadata 
        del self.cited_paper
        del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationTextGenerationRAGDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 #cited_paragraph_ids_path="cited_text_embeddings/cited_paragraph_ids.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 reference_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                 cited_paper_ids = None,
                 context_window_size = None,
                 rouge_oracle_retrieved_paragraphs = None,
                 n_docs = 5,
                 ):

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)
            
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        if rouge_oracle_retrieved_paragraphs is not None:
            ranked_ROUGE_cited_indices = {}
            with open(rouge_oracle_retrieved_paragraphs) as f:
                for line in f:
                    rouge_dict = json.loads(line)
                    ranked_ROUGE_cited_indices[rouge_dict["id"]] = rouge_dict

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        #self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        """
        with open(cited_paragraph_ids_path) as f:
            cited_paragraph_ids_list = json.load(f)

        candidate_paragraph_indices = {}
        for i, pid in enumerate(cited_paragraph_ids_list["ids"]):
            link, _ = pid.split("_")
            this_link = candidate_paragraph_indices.get(link, [])
            this_link.append(i)
            candidate_paragraph_indices[link] = this_link
        """

        self.dataset = []
        self.samples = []
        
        self.lens = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try: 
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, 16384) ####################### Do not do segmentation for now.

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    sent_span_mapping = get_sent_span_mapping(span_sent_mapping, len(span_citation_mapping))
                    sentence_bounds = get_sentence_bounds(paragraph, offset_mapping)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        #"offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info,
                        "sentence_bounds": sentence_bounds,
                        "sent_span_mapping": sent_span_mapping,
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        if dominant_only and span["span_type"] == "Reference":
                            continue
                        elif reference_only and span["span_type"] == "Dominant":
                            continue
                        if context_window_size is not None:
                            context_char_start, context_char_end = context_char_bound(
                                    i_span, 
                                    sentence_bounds, 
                                    span_sent_mapping, 
                                    sent_span_mapping, 
                                    context_window_size
                                )
                        else:
                            context_char_start = 0
                            context_char_end = -1
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)                                
                        context_before = paragraph[context_char_start:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:context_char_end].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")
                        if include_context:
                            source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.extend(["[Dominant]", tokenizer.mask_token])
                        else:
                            source.extend(["[Reference]", tokenizer.mask_token])
                        if not auto_regressive:
                            source.append(context_after)
                        
                        self.lens.append(len(tokenizer.tokenize(" ".join(source))))

                        citation_marks = []
                        citation_links = set([])
                        span_id = paragraph_id + "_" + str(i_span)
                        for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                                            ["[B_Dominant]", "[B_Reference]"],
                                                            ["[E_Dominant]", "[E_Reference]"]):
                            for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                if link is not None and (cited_paper_ids is None or link in cited_paper_ids):
                                    # To avoid being truncated by RAG, prepend citation marks
                                    source = [b_span, citation_mark, e_span] + source
                                    #source.append(b_span)
                                    #source.append(citation_mark)
                                    citation_marks.append(citation_mark)
                                    citation_links.add(span_type[0]+link)
                                    #source.append(e_span)
                                    span_id += "@"+link
                                
                        source = " ".join(source)

                        #allowed_doc_ids = []
                        #all_found = True
                        #for link in citation_links:
                        #    if link in candidate_paragraph_indices:
                        #        allowed_doc_ids.extend(candidate_paragraph_indices[link])
                        #    else:
                        #        all_found = False

                        if len(citation_links) > 0 and len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            if rouge_oracle_retrieved_paragraphs is not None:
                                if span_id not in ranked_ROUGE_cited_indices:
                                    continue
                                retrieved_doc_ids = ranked_ROUGE_cited_indices[span_id]["cited_indices"][:n_docs]
                                retrieved_doc_ids = "@".join([str(i) for i in retrieved_doc_ids])
                            else:
                                retrieved_doc_ids = "0"
                            self.samples.append({
                                "id": span_id,
                                "source": source,
                                "target": target,
                                "citations": "#".join(citation_marks),
                                "citation_links": "@".join(list(citation_links)),
                                "retrieved_doc_ids": retrieved_doc_ids,
                                #"allowed_doc_ids": allowed_doc_ids,
                            })

            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        #del self.cited_metadata 
        del self.cited_paper
        del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CTS_DPR_Dataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 #cited_paragraph_ids_path="cited_text_embeddings/cited_paragraph_ids.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                 context_window_size = 2,
                 sentence_level = False,
                 ):

        stops = set(stopwords.words('english'))

        def get_citation_mark(metadata):
            if len(metadata["authors"]) > 0:
                first_author = metadata["authors"][0]["last"]
            else:
                first_author = ""
            return first_author + "@" + str(metadata.get("year",""))

        def remove_citation_marks(text, citations):
            for citation in citations:
                text = text.replace(citation,"")
            return text

        def remove_stop_words(sentence):
            tokens = word_tokenize(sentence)
            cleaned_tokens = []
            for token in tokens:
                if token not in stops:
                    cleaned_tokens.append(token)
            return " ".join(cleaned_tokens)

        def compute_similarity(candidate, target, citations):
            scores = scorer.score(remove_stop_words(remove_citation_marks(target, citations)), remove_stop_words(candidate))
            return round(scores["rougeL"].recall, 4)

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)
            
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        """
        with open(cited_paragraph_ids_path) as f:
            cited_paragraph_ids_list = json.load(f)

        candidate_paragraph_indices = {}
        for i, pid in enumerate(cited_paragraph_ids_list["ids"]):
            link, _ = pid.split("_")
            this_link = candidate_paragraph_indices.get(link, [])
            this_link.append(i)
            candidate_paragraph_indices[link] = this_link
        """

        self.dataset = []
        self.samples = []
        
        self.lens = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, 16384) ####################### Do not do segmentation for now.

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    sent_span_mapping = get_sent_span_mapping(span_sent_mapping, len(span_citation_mapping))
                    sentence_bounds = get_sentence_bounds(paragraph, offset_mapping)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        #"offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info,
                        "sentence_bounds": sentence_bounds,
                        "sent_span_mapping": sent_span_mapping,
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        if not dominant_only or span["span_type"] == "Dominant":
                            if context_window_size is not None:
                                context_char_start, context_char_end = context_char_bound(
                                        i_span, 
                                        sentence_bounds, 
                                        span_sent_mapping, 
                                        sent_span_mapping, 
                                        context_window_size
                                    )
                            else:
                                context_char_start = 0
                                context_char_end = -1
                            source = []
                            if include_intro:
                                source.append(introduction_text)
                                source.append(tokenizer.sep_token)                                
                            context_before = paragraph[context_char_start:span["char_start"]].replace(
                                "[BOS] ", "")
                            context_after = paragraph[span["char_end"]:context_char_end].replace(
                                "[BOS] ", "")
                            target = paragraph[
                                    span["char_start"]:span["char_end"]].replace(
                                "[BOS] ", "")
                            if include_context:
                                source.append(context_before)
                            if len(span["span_citation_mapping"]["Dominant"]) > 0:
                                source.extend(["[Dominant]", tokenizer.mask_token])
                            else:
                                source.extend(["[Reference]", tokenizer.mask_token])
                            if not auto_regressive:
                                source.append(context_after)
                            
                            self.lens.append(len(tokenizer.tokenize(" ".join(source))))

                            citation_marks = []
                            citation_links = set([])
                            span_id = paragraph_id + "_" + str(i_span)
                            for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                                                ["[B_Dominant]", "[B_Reference]"],
                                                                ["[E_Dominant]", "[E_Reference]"]):
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    if link:
                                        source = [b_span, citation_mark, e_span] + source
                                        #source.append(b_span)
                                        #source.append(citation_mark)
                                        citation_marks.append(citation_mark)
                                        citation_links.add(link)
                                        #source.append(e_span)
                                        span_id += "@"+link
                                    
                            source = " ".join(source)

                            #allowed_doc_ids = []
                            #all_found = True
                            #for link in citation_links:
                            #    if link in candidate_paragraph_indices:
                            #        allowed_doc_ids.extend(candidate_paragraph_indices[link])
                            #    else:
                            #        all_found = False

                            cited_paragraphs = {}

                            for link in citation_links:
                                if link and link in self.cited_paper:
                                    this_cited_paper = []

                                    metadata = self.cited_metadata[link]
                                    citation_mark = get_citation_mark(metadata)
                                    cited_paper = self.cited_paper[link]
                                    
                                    if sentence_level:
                                        cited_paragraph = {
                                            "id": cited_paper["paper_id"] + "_t0",
                                            "title": citation_mark,
                                            "text": "Title: " + cited_paper["title"],
                                            "score": compute_similarity(cited_paper["title"], target, citation_marks)
                                        }

                                        if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                                            this_cited_paper.append(cited_paragraph)
                                        
                                        for i, paragraph_obj in enumerate(cited_paper["abstract"]):
                                            segmented_sentences = patch_sent_tokenize(sent_tokenize(paragraph_obj["text"]))
                                            for si, sent in enumerate(segmented_sentences):
                                                cited_paragraph = {
                                                    "id": cited_paper["paper_id"] + "_a" + str(i) + "#" + str(si),
                                                    "title": citation_mark,
                                                    "text":  paragraph_obj["section"] + ": " + sent,
                                                    "score": compute_similarity(sent, target, citation_marks)
                                                }
                                                if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                                                    this_cited_paper.append(cited_paragraph)
                                                
                                        for i, paragraph_obj in enumerate(cited_paper["body_text"]):
                                            segmented_sentences = patch_sent_tokenize(sent_tokenize(paragraph_obj["text"]))
                                            for si, sent in enumerate(segmented_sentences):
                                                cited_paragraph = {
                                                    "id": cited_paper["paper_id"] + "_b" + str(i) + "#" + str(si),
                                                    "title": citation_mark,
                                                    "text":  paragraph_obj["section"] + ": " + sent,
                                                    "score": compute_similarity(sent, target, citation_marks)
                                                }
                                                if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                                                    this_cited_paper.append(cited_paragraph)

                                        cited_paragraphs[link] = this_cited_paper
                                    
                                    else:

                                        cited_paragraph = {
                                            "id": cited_paper["paper_id"] + "_t0",
                                            "title": citation_mark,
                                            "text": "Title: " + cited_paper["title"],
                                            "score": compute_similarity(cited_paper["title"], target, citation_marks)
                                        }

                                        if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                                            this_cited_paper.append(cited_paragraph)
                                        
                                        for i, paragraph_obj in enumerate(cited_paper["abstract"]):
                                            cited_paragraph = {
                                                "id": cited_paper["paper_id"] + "_a" + str(i),
                                                "title": citation_mark,
                                                "text": paragraph_obj["section"] + ": " + paragraph_obj["text"],
                                                "score": compute_similarity(paragraph_obj["text"], target, citation_marks)
                                            }
                                            if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                                                this_cited_paper.append(cited_paragraph)
                                                
                                        for i, paragraph_obj in enumerate(cited_paper["body_text"]):
                                            cited_paragraph = {
                                                "id": cited_paper["paper_id"] + "_b" + str(i),
                                                "title": citation_mark,
                                                "text": paragraph_obj["section"] + ": " + paragraph_obj["text"],
                                                "score": compute_similarity(paragraph_obj["text"], target, citation_marks)
                                            }
                                            if len(cited_paragraph["title"]) > 0 and len(cited_paragraph["text"]) > 0:
                                                this_cited_paper.append(cited_paragraph)

                                        cited_paragraphs[link] = this_cited_paper


                            if len(citation_links) > 0 and len(tokenizer.tokenize(source)) <= self.max_sent_len:
                                self.samples.append({
                                    "id": span_id,
                                    "source": source,
                                    "target": target,
                                    "citations": "#".join(citation_marks),
                                    "citation_links": "@".join(list(citation_links)),
                                    "cited_paragraphs": cited_paragraphs
                                })

            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        del self.cited_metadata 
        del self.cited_paper
        del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def make_dpr_inputs(self, dataset_name="", top_k=20, n_negative=100):
        all_cited_papers = {}
        for data in self.samples:
            for link, cited_paragraphs in data["cited_paragraphs"].items():
                for paragraph in cited_paragraphs:
                    all_cited_papers[paragraph["id"]] = {
                        "title": paragraph["title"],
                        "text": paragraph["text"],
                        "id": paragraph["id"]
                    }

        all_pairs = []
        for data in tqdm(self.samples):
            covered_paragraphs = []
            positive_paragraphs = []
            hard_negative_paragraphs = []
            for link, cited_paragraphs in data["cited_paragraphs"].items():
                covered_paragraphs.extend([paragraph["id"] for paragraph in cited_paragraphs])
                sorted_cited_paragraphs = sorted(cited_paragraphs, key= lambda x: x["score"], reverse=True)
                positive_paragraphs.extend(sorted_cited_paragraphs[:top_k])
                hard_negative_paragraphs.extend(sorted_cited_paragraphs[top_k:])
            #covered_paragraphs = set(covered_paragraphs)
            #negative_ids = list(set(all_cited_papers.keys()) - covered_paragraphs)
            #sampled_negative_ids = random.sample(negative_ids, min([n_negative, len(negative_ids)]))
            #negative_paragraphs = [all_cited_papers[Id] for Id in sampled_negative_ids]
            
            this_pair = {
                "dataset": dataset_name, 
                "question": data["source"],
                "answers": [data["target"]],
                "positive_ctxs": [],
                "negative_ctxs": [],
                "hard_negative_ctxs": [],
            }
            for paragraph in positive_paragraphs:
                this_pair["positive_ctxs"].append({
                    "title": paragraph["title"],
                    "text": paragraph["text"],
                    "score": paragraph["score"],
                    "title_score": 1,
                    "passage_id": paragraph["id"]
                })
            for paragraph in hard_negative_paragraphs:
                this_pair["hard_negative_ctxs"].append({
                    "title": paragraph["title"],
                    "text": paragraph["text"],
                    "score": paragraph["score"],
                    "title_score": 0,
                    "passage_id": paragraph["id"]
                })
            #for paragraph in negative_paragraphs:
            #    this_pair["hard_negative_ctxs"].append({
            #        "title": paragraph["title"],
            #        "text": paragraph["text"],
            #        "score": 0,
            #        "title_score": 0,
            #        "passage_id": paragraph["id"]
            #    })
            all_pairs.append(this_pair)
        return all_pairs

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset
    
class SimpleShortCrossDocumentLMdataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True,
                 MAX_SENT_LEN=1024,
                 mask_token="<mask>",
                 bod_token="<doc>", eod_token="</doc>",
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl'):
        self.max_sent_len = MAX_SENT_LEN
        self.related_work_jsons = read_related_work_jsons(path_name)
        self.cited_metadata_jsons = read_related_work_jsons(cited_metadata_path)

        self.samples = []

        for i, (ID, related_work) in tqdm(
                enumerate(self.related_work_jsons.items())):
            year = related_work["year"]
            if year is None:
                year = 0
            if (train and year <= 2017) or (not train and year == 2018):
                bib_entries = related_work["bib_entries"]
                for paragraph in related_work["related_work"]:
                    inputs = []
                    noisy_text, target = makeMLMsample(paragraph["text"], mask_token = mask_token)
                    inputs.extend([bod_token, noisy_text, eod_token])
                    if len(tokenizer(target)["input_ids"]) > self.max_sent_len:
                        continue

                    for citation in paragraph["cite_spans"]:
                        if citation["ref_id"] in bib_entries:
                            this_inputs = deepcopy(inputs)
                            reference_link = bib_entries[citation["ref_id"]][
                                "link"]
                            if reference_link in self.cited_metadata_jsons:
                                cited_metadata = self.cited_metadata_jsons[
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
                                           "input_ids"]) > self.max_sent_len:
                                    continue
                                source = " ".join(this_inputs)
                                self.samples.append({
                                    "id": ID + "_" + str(i) + "_" + str(reference_link),
                                    "source": source,
                                    "target": target
                                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class SimpleShortCrossDocumentLMparallelDataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True,
                 MAX_SENT_LEN=1024,
                 bod_token="<doc>", eod_token="</doc>",
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 n_process = 100
                ):
        self.max_sent_len = MAX_SENT_LEN
        self.bod_token = bod_token
        self.eod_token = eod_token
        self.train = train
        self.related_work_jsons = read_related_work_jsons(path_name)
        self.cited_metadata_jsons = read_related_work_jsons(cited_metadata_path)
        self.tokenizer = tokenizer
        self.samples = []
        print("Begin assigning!")
        p = Pool(processes=n_process)
        distributed_related_work_jsons = [{} for i in range(n_process)]
        for i, (ID, related_work) in enumerate(self.related_work_jsons.items()):
            which_p = i % n_process
            distributed_related_work_jsons[which_p][ID] = related_work
        print("Begin working!")
        samples = p.map(self.job, distributed_related_work_jsons)
        p.close()
        for s in samples:
            self.samples.extend(s)
        del self.tokenizer
        
    def job(self, related_work_jsons):
        local_samples = []
        for ID, related_work in tqdm(related_work_jsons.items()):
            year = related_work["year"]
            if year is None:
                year = 0
            if (self.train and year <= 2017) or (not self.train and year == 2018):
                bib_entries = related_work["bib_entries"]
                for i, paragraph in enumerate(related_work["related_work"]):
                    inputs = []
                    noisy_text, target = makeMLMsample(paragraph["text"], mask_token = self.tokenizer.mask_token)
                    inputs.extend([self.bod_token, noisy_text, self.eod_token])
                    for citation in paragraph["cite_spans"]:
                        if citation["ref_id"] in bib_entries:
                            this_inputs = deepcopy(inputs)
                            reference_link = bib_entries[citation["ref_id"]][
                                "link"]
                            if reference_link in self.cited_metadata_jsons:
                                cited_metadata = self.cited_metadata_jsons[
                                    reference_link]
                                title = cited_metadata["title"]
                                if title is None:
                                    title = ""
                                abstract = cited_metadata["abstract"]
                                if abstract is None:
                                    abstract = ""
                                this_inputs.extend(
                                    [self.bod_token, title, self.tokenizer.sep_token,
                                     abstract, self.eod_token])
                                if len(self.tokenizer(" ".join(this_inputs))[
                                           "input_ids"]) > self.max_sent_len:
                                    continue
                                source = " ".join(this_inputs)
                                local_samples.append({
                                    "id": ID + "_" + str(i) + "_" + str(reference_link),
                                    "source": source,
                                    "target": target
                                })
        return local_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class SingleAbstractCitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                 ):
        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)
            
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        #self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        
        self.lens = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try: 
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        # "offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        if not dominant_only or span["span_type"] == "Dominant":
                            source = []
                            if include_intro:
                                source.append(introduction_text)
                                source.append(tokenizer.sep_token)                                
                            context_before = paragraph[:span["char_start"]].replace(
                                "[BOS] ", "")
                            context_after = paragraph[span["char_end"]:].replace(
                                "[BOS] ", "")
                            target = paragraph[
                                    span["char_start"]:span["char_end"]].replace(
                                "[BOS] ", "")
                            if include_context:
                                source.append(context_before)
                            if len(span["span_citation_mapping"]["Dominant"]) > 0:
                                source.extend(["[Dominant]", tokenizer.mask_token])
                            else:
                                source.extend(["[Reference]", tokenizer.mask_token])
                            if not auto_regressive:
                                source.append(context_after)
                            
                            self.lens.append(len(tokenizer.tokenize(" ".join(source))))
                            
                            citation_types = {}
                            cited_paragraphs = OrderedDict()
                            citation_marks_flag = False
                            #high_quality_span = True

                            for span_type in ["Dominant", "Reference"]:
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    citation_marks_flag = True
                                    if link and link in self.cited_paper:
                                        citation_types[link] = span_type
                                        abstract, high_quality_abstract = get_title_abstract(self.cited_paper[link],
                                                                include_result=include_conclusion)
                                        cited_paragraphs[link] = abstract

                            if skip_no_citations and not citation_marks_flag:
                                continue

                            for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                                                ["[B_Dominant]", "[B_Reference]"],
                                                                ["[E_Dominant]", "[E_Reference]"]):
                                for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                    citation_marks = []
                                    this_source = deepcopy(source)
                                    this_source = [b_span, citation_mark, tokenizer.sep_token, truncate_string(cited_paragraphs[link], tokenizer, self.max_abstract_length), e_span, tokenizer.sep_token] + this_source
                                    citation_marks.append(citation_mark)

                                    this_source = " ".join(this_source)

                                    if len(tokenizer.tokenize(this_source)) <= self.max_sent_len:
                                        span_id = paragraph_id + "_" + str(i_span)
                                        for link in cited_paragraphs:
                                            span_id += "@"+citation_types[link][0]+"_"+link

                                        self.samples.append({
                                            "id": span_id,
                                            "source": this_source,
                                            "target": target,
                                            "citations": "#".join(citation_marks)
                                        })

            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        #del self.cited_metadata 
        del self.cited_paper
        #del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitationTextGenerationRetrievedCTSDataset(Dataset):
    def __init__(self, path_name: str, 
                retrieved_paragraphs, 
                cited_dataset,
                tokenizer, train=True, MAX_SENT_LEN=99999,
                 max_abstract_length = 400,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=False,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 auto_regressive=False,
                 dominant_only=False,
                 reference_only=False,
                 start = 0,
                 limit = None,
                 include_context = True,
                 context_window_size = None,
                 n_docs = 5,
                 add_keywords = False,
                 ):

        cited_paper_ids = set(cited_dataset["paper_id"])

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)
            
            high_quality = True if paper_dict["title"] and len(paras)>0 else False            
            return paper_dict["title"] + " | " + " ".join(paras), high_quality

        self.max_sent_len = MAX_SENT_LEN
        self.max_abstract_length = max_abstract_length
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        ranked_cited_indices = {}
        with open(retrieved_paragraphs) as f:
            for line in f:
                rouge_dict = json.loads(line)
                ranked_cited_indices[rouge_dict["id"]] = rouge_dict

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        #self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        
        self.lens = []
        
        if limit:
            text_files = text_files[start:limit]
        else:
            text_files = text_files[start:]
        
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            citing_paper_id = tail.split(".")[0]
            paper_id = tail.split(".")[0]
            try: 
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, 16384) ####################### Do not do segmentation for now.

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text, _ = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                            annotation[
                                -1]
                    #try:
                    tokens = tokenizer.tokenize(paragraph,
                                                add_special_tokens=True)
                    # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                    sentences = [sent for sent in
                                paragraph.split("[BOS] ")[1:]]

                    offset_mapping = \
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"]
                    N_tokens = len(offset_mapping)
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation,
                        paragraph,
                        self.discourse_label_types)
                    span_indices = read_span_indices(paragraph_annotation,
                                                    paragraph)
                    span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                            offset_mapping)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_aligned_BIO_labels(
                        citation_mark_span_indices, offset_mapping)
                    # print(tokenizer.tokenize(paragraph))
                    assert (N_tokens == len(span_BIO_labels) == len(
                        citation_BIO_labels))
                    # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                    #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                    #    continue

                    # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                    # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                    pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                        paragraph_id, sentences, self.related_work_jsons,
                        offset_mapping, citation_BIO_labels,
                        separator="[BOS] ")
                    paragraph_citation_links_pre = new_sentence_citation_link(
                        pargraph_citation_info, len(sentences))
                    # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                    span_citation_mapping = map_span_citation(
                        span_BIO_labels,
                        citation_BIO_labels,
                        pargraph_citation_info,
                        offset_mapping)
                    span_sent_mapping, i_span = new_span_sentence_map(
                        tokens,
                        span_BIO_labels,
                        bos="[BOS]")
                    paragraph_citation_links = propagate_citation_cross_sentences(
                        span_sent_mapping, paragraph_citation_links_pre,
                        i_span)
                    sent_span_mapping = get_sent_span_mapping(span_sent_mapping, len(span_citation_mapping))
                    sentence_bounds = get_sentence_bounds(paragraph, offset_mapping)
                    self.dataset.append({
                        "paragraph_id": paragraph_id,
                        "paragraph": paragraph,
                        # "related_work": augmented_paragraph,
                        "citation_links_by_sentence": paragraph_citation_links,
                        # "augmented_sentences": augmented_sentences,
                        "discourse_labels": discourse_labels,
                        "sentences": sentences,
                        # "span_labels": span_BIO_labels,
                        # "citation_labels": citation_BIO_labels,
                        "span_sent_mapping": span_sent_mapping,
                        # "tokens": tokens
                        # "i_span": i_span,
                        "span_citation_mapping": span_citation_mapping,
                        #"offset_mapping": offset_mapping,
                        "citation_info": pargraph_citation_info,
                        "sentence_bounds": sentence_bounds,
                        "sent_span_mapping": sent_span_mapping,
                    })
                    #except:
                    #    continue
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        if dominant_only and span["span_type"] == "Reference":
                            continue
                        elif reference_only and span["span_type"] == "Dominant":
                            continue
                        if context_window_size is not None:
                            context_char_start, context_char_end = context_char_bound(
                                    i_span, 
                                    sentence_bounds, 
                                    span_sent_mapping, 
                                    sent_span_mapping, 
                                    context_window_size
                                )
                        else:
                            context_char_start = 0
                            context_char_end = -1
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)                                
                        context_before = paragraph[context_char_start:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:context_char_end].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")
                        if include_context:
                            source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.extend(["[Dominant]", tokenizer.mask_token])
                        else:
                            source.extend(["[Reference]", tokenizer.mask_token])
                        if not auto_regressive:
                            source.append(context_after)

                        span_id = paragraph_id + "_" + str(i_span)
                        
                        citation_marks = []
                        citation_links = set([])
                        
                        for span_type, b_span, e_span in zip(["Dominant", "Reference"], 
                                ["[B_Dominant]", "[B_Reference]"],
                                ["[E_Dominant]", "[E_Reference]"]):
                            for citation_mark, link in span["span_citation_mapping"][span_type].items():
                                if link is not None and (cited_paper_ids is None or link in cited_paper_ids):
                                    # To avoid being truncated by RAG, prepend citation marks
                                    source = [b_span, citation_mark, e_span] + source
                                    #source.append(b_span)
                                    #source.append(citation_mark)
                                    citation_marks.append(citation_mark)
                                    citation_links.add(span_type[0]+link)
                                    #source.append(e_span)
                                    span_id += "@"+link
                                
                        if len(citation_links) > 0:
                            if span_id not in ranked_cited_indices:
                                continue
                            if add_keywords:
                                keywords = ranked_cited_indices[span_id]["keywords"]
                                source.extend([" ".join(keywords)])

                            retrieved_doc_ids = ranked_cited_indices[span_id]["cited_indices"][:n_docs]
                            for doc_id in retrieved_doc_ids:
                                if doc_id >= 0:
                                    retireved = cited_dataset[doc_id]
                                    source.extend([tokenizer.sep_token, retireved["title"], " // ", retireved["text"]])
                            source = " ".join(source)
                            if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                                self.samples.append({
                                    "id": span_id,
                                    "source": source,
                                    "target": target,
                                    "citations": "#".join(citation_marks),
                                    "citation_links": "@".join(list(citation_links)),
                                    #"retrieved_doc_ids": retrieved_doc_ids,
                                    #"allowed_doc_ids": allowed_doc_ids,
                                })

            except:
                #print("Skip "+paper_id)
                continue

        del self.citing_paper 
        del self.related_work_jsons 
        #del self.cited_metadata 
        del self.cited_paper
        del self.dataset
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitationTextGenerationAnnotatedCTSDataset(Dataset):
    def __init__(self, path_name, 
                tokenizer, 
                auto_regressive=False,
                context_window_size = 2,
                annotated_CTS = True,
                n_docs=10,
                ):

        with open(path_name) as f:
            CTS_by_citation = json.load(f)

        self.samples = []
        for citation_id,citation in CTS_by_citation.items():
            source = ""
            
            for i in range(context_window_size*-1,0,1):
                if i in citation["context"]:
                    source += citation["context"][i] + "\n "
            source += " [Dominant] "+ tokenizer.mask_token + " "
            if not auto_regressive:
                for i in range(1, context_window_size+1):
                    if i in citation["context"]:
                        source += citation["context"][i] + "\n "
            if annotated_CTS:
                citation_marks = []
                for cited_paper_id, cited_paper in citation["cited_papers"].items():
                    citation_marks.append(cited_paper["citation_marker"])
                    for sentence in cited_paper["support_sentences"]:
                        source += " "+tokenizer.sep_token+" "+ cited_paper["citation_marker"] + " // " + sentence
            
            else:
                citation_marks = set([])
                for sentence, citation_mark in zip(citation["sorted_support_sentences"][:n_docs], 
                                                citation["sorted_support_sentences_paper_ids"][:n_docs]):
                    source += " "+ tokenizer.sep_token+" "+ citation_mark + " // " + sentence
                    citation_marks.add(citation_mark)
                citation_marks = list(citation_marks)
                
            target = citation["citing_sentence"]
            self.samples.append({
                "id": citation_id,
                "source": source,
                "target": target,
                "citations": "#".join(citation_marks)
            })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitationTextGenerationSciSummDataset(Dataset):
    def __init__(self, path_name, 
                tokenizer, 
                auto_regressive=False,
                context_window_size = 2,
                annotated_CTS = True,
                n_docs=10,
                agreed_only = False,
                ):
        use_annotated_CTS = annotated_CTS
        with open(path_name) as f:
            cl_scisumm_citation_text_dataset = json.load(f)

        self.samples = []
        for cited_paper_name, citations in cl_scisumm_citation_text_dataset.items():
            for citation_idx, citation in citations.items():
                citation_id = cited_paper_name+"_"+str(citation_idx)
                source = ""
                for i in range(context_window_size*-1,0,1):
                    if i in citation["context"]:
                        source += citation["context"][i] + "\n "
                source += " [Dominant] "+ tokenizer.mask_token + " "
                if not auto_regressive:
                    for i in range(1, context_window_size+1):
                        if i in citation["context"]:
                            source += citation["context"][i] + "\n "
                if use_annotated_CTS:
                    citation_marks = [citation["citation_mark"]]
                    if agreed_only:
                        annotated_CTS = []
                        for author, CTS in citation["annotated_CTS"].items():
                            annotated_CTS.append(CTS)
                        agreed_CTS = set(annotated_CTS[0])
                        for CTS in annotated_CTS:
                            agreed_CTS = agreed_CTS.intersection(set(CTS))
                        annotated_CTS = list(agreed_CTS)
                        if len(annotated_CTS) == 0:
                            continue
                    else:
                        annotated_CTS = []
                        for author, CTS in citation["annotated_CTS"].items():
                            annotated_CTS.extend(CTS)
                        annotated_CTS = list(set(annotated_CTS))
                    
                    for sentence in annotated_CTS:
                        source += " "+tokenizer.sep_token+" "+ citation["citation_mark"] + " // " + sentence
                        
                else:
                    citation_marks = [citation["citation_mark"]]
                    for sentence in citation["rouge_ranked_CTS"][:n_docs]:
                        source += " "+ tokenizer.sep_token+" "+ citation["citation_mark"] + " // " + sentence

                target = citation["citation_text"]
                self.samples.append({
                    "id": citation_id,
                    "source": source,
                    "target": target,
                    "citations": "#".join(citation_marks)
                })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        #self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset