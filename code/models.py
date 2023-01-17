import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, BartForConditionalGeneration, AutoConfig, RagConfig#, RagRetriever#, RagSequenceForGeneration, RagTokenForGeneration
from retrieval_rag import RagRetriever
from modeling_rag import RagSequenceForGeneration, RagTokenForGeneration
from modeling_FiDBart import FiDBart
#from modeling_graph_t5 import T5ForConditionalGeneration

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class TimeDistributedDense(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(TimeDistributedDense, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.timedistributedlayer = TimeDistributed(self.linear)
    def forward(self, x):
        # x: (BATCH_SIZE, ARRAY_LEN, INPUT_SIZE)
        
        return self.timedistributedlayer(x)
    
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob = 0.1):
        super().__init__()
        self.dense = TimeDistributedDense(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = TimeDistributedDense(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CitedTextSpanSelector(nn.Module):
    def __init__(self, bert_path, tokenizer_len, CTS_token_id, dropout = 0):
        super(CitedTextSpanSelector, self).__init__()
        self.CTS_token_id = CTS_token_id
        seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
            bert_path
        )
        seq2seq.resize_token_embeddings(tokenizer_len)
        self.bert = seq2seq.get_encoder()
        
        self.bert_dim = self.bert.config.hidden_size # bert_dim
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ranking_loss = nn.MarginRankingLoss(reduction="mean", margin=0)
        self.dropout = dropout
        self.linear = ClassificationHead(self.bert_dim, 1, hidden_dropout_prob = dropout)
        self.extra_modules = [
            self.linear,
            self.kl_div,
            self.ranking_loss
        ]
    
    def forward(self, encoded_dict, labels = None):
        if hasattr(self,"module"):
            bert_out = self.module.bert(**encoded_dict)[0]
        else:
            bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)     
            
        out = self.linear(bert_out)
        boolean_mask = encoded_dict["input_ids"] == self.CTS_token_id
        #mask = boolean_mask.long().unsqueeze(-1)      
        cts_scores = out.view(-1)[boolean_mask.view(-1)]
        #cts_scores = out.view(-1,2)[boolean_mask.view(-1),:]
        ##max = torch.max(cts_scores)
        #min = torch.min(cts_scores)
        #cts_scores = (cts_scores - min) / (max - min + 1e-5)
        #cts_scores = torch.sigmoid(cts_scores)
        cts_scores = torch.softmax(cts_scores,-1)
        log_cts_scores = torch.log_softmax(cts_scores,-1)
        if labels is not None:
            cts_labels = labels.view(-1)[boolean_mask.view(-1)]
            kld_loss = self.kl_div(log_cts_scores.unsqueeze(0), cts_labels.unsqueeze(0)) #/ torch.sum(mask).item()
            ranking_loss = self.compute_ranking_loss(cts_scores, cts_labels)
            loss = ranking_loss + kld_loss
        else:
            loss = None
            
        #pred = torch.argmax(out.cpu(), dim=-1) # (Batch_size, N_sep)
        #out = [pred_paragraph[:n_token].detach().numpy().tolist() for pred_paragraph, n_token in zip(pred, N_tokens)] 
        return cts_scores, loss

    def compute_ranking_loss(self, prediction, labels):
        input1 = []
        input2 = []
        target = []
        for i in range(prediction.shape[0]):
            for j in range(i+1, prediction.shape[0]):
                input1.append(prediction[i].unsqueeze(0))
                input2.append(prediction[j].unsqueeze(0))
                target.append((labels[i] > labels[j]).unsqueeze(0))
        
        return self.ranking_loss(torch.cat(input1), torch.cat(input2), torch.cat(target) * 2 - 1)

class OracleRetriever:
    def __init__(self, cited_dataset, question_encoder_tokenizer, generator_tokenizer, max_combined_length):
        self.max_combined_length = max_combined_length
        self.dataset = cited_dataset
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.generator_tokenizer = generator_tokenizer

    def get_doc_dicts(self, doc_ids: np.ndarray):
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def retrieve(self, doc_ids):
        docs = self.get_doc_dicts(doc_ids)
        retrieved_doc_embeds = [doc["embeddings"] for doc in docs] #(batch_size, n_docs, d)
        return retrieved_doc_embeds, docs

    def __call__(self, input_ids, doc_ids, prefix=None, n_docs=None, return_tensors=None):
        n_docs = doc_ids.shape[1] if n_docs is None else n_docs
        retrieved_doc_embeds, docs = self.retrieve(doc_ids)
        input_strings = self.question_encoder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )
        return BatchEncoding( 
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": retrieved_doc_embeds,
                "doc_ids": doc_ids,
            },
            tensor_type=return_tensors,
        )

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        title_sep = " / "
        doc_sep = " // "
        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + doc_title + title_sep + doc_text + doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.max_combined_length,
            return_tensors=return_tensors,
            padding="longest",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

class CitationSpanGeneratorRAG(nn.Module):
    def __init__(self, rag_path, question_encoder_path, generator_path, retriever, oracle_retriever = None, n_docs=10, question_encoder_tokenizer_len = None, generator_tokenizer_len = None, use_fid=False, cited_abstract_only=False, reference_abstract_only=True):
        super(CitationSpanGeneratorRAG, self).__init__()
        self.n_docs = n_docs
        self.cited_abstract_only = cited_abstract_only
        self.reference_abstract_only = reference_abstract_only 
        self.retriever = retriever
        self.oracle_retriever = oracle_retriever
        
        rag_config = RagConfig.from_pretrained(rag_path)
        gen_config = AutoConfig.from_pretrained(generator_path)
        question_encoder_config = AutoConfig.from_pretrained(question_encoder_path)

        rag_config.generator = gen_config
        rag_config.question_encoder = question_encoder_config

        if use_fid:
            generator_model = FiDBart.from_pretrained(generator_path)
        else:
            generator_model = BartForConditionalGeneration.from_pretrained(generator_path)
        
        if "rag-token" in rag_path:
            self.model = RagTokenForGeneration.from_pretrained_question_encoder_generator(
                question_encoder_path, generator_path, config=rag_config, generator_model=generator_model
            )
        else:
            self.model = RagSequenceForGeneration.from_pretrained_question_encoder_generator(
                question_encoder_path, generator_path, config=rag_config, generator_model=generator_model
            )
        
        if use_fid:
            self.model.generator.encoder.n_passages = self.n_docs
        #self.model = RagTokenForGeneration.from_pretrained(model_path, retriever=self.retriever)
        #self.model = RagSequenceForGeneration.from_pretrained(model_path, retriever=self.retriever)
        self.model.config.n_docs = self.n_docs

        if question_encoder_tokenizer_len:
            self.model.question_encoder.resize_token_embeddings(question_encoder_tokenizer_len)
        if generator_tokenizer_len:
            self.model.generator.resize_token_embeddings(generator_tokenizer_len)
        
    def _forward(self, input_ids = None, attention_mask = None, allowed_doc_ids=None, oracle_doc_ids=None, use_oracle=False):
        question_hidden_states = self.model.question_encoder(input_ids = input_ids, attention_mask = attention_mask).pooler_output
        if use_oracle:
            docs_dict = self.oracle_retriever(input_ids.cpu().numpy(), oracle_doc_ids.detach().cpu().numpy(), return_tensors="pt", n_docs=self.n_docs)
        else:
            docs_dict = self.retriever(input_ids.cpu().numpy(), question_hidden_states.detach().cpu().numpy(), allowed_doc_ids.detach().cpu().tolist(), return_tensors="pt", n_docs=self.n_docs, cited_abstract_only = self.cited_abstract_only, reference_abstract_only = self.reference_abstract_only)

        filtered_docs_dict = {
            "context_input_ids": docs_dict["context_input_ids"].to(question_hidden_states).long(),
            "context_attention_mask": docs_dict["context_attention_mask"].to(question_hidden_states).long(),
            "retrieved_doc_embeds": docs_dict["retrieved_doc_embeds"].to(question_hidden_states),
        }
        
        doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1), filtered_docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1)
        return filtered_docs_dict, doc_scores, docs_dict["doc_ids"]

    def forward(self, input_ids = None, attention_mask = None, allowed_doc_ids=None, labels = None, oracle_doc_ids = None, use_oracle = False, **model_kwargs):
        filtered_docs_dict, doc_scores, _ = self._forward(input_ids, attention_mask, allowed_doc_ids, oracle_doc_ids, use_oracle)
        outputs = self.model(
            context_input_ids=filtered_docs_dict["context_input_ids"],
            context_attention_mask=filtered_docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            decoder_input_ids=labels,
            n_docs=self.n_docs,
            labels = labels,
        )
        return outputs
    
    def generate(self, input_ids = None, attention_mask = None, allowed_doc_ids=None, oracle_doc_ids=None, use_oracle = False, **gen_kwargs):
        filtered_docs_dict, doc_scores, doc_ids = self._forward(input_ids, attention_mask, allowed_doc_ids, oracle_doc_ids, use_oracle)
        #print("context_input_ids", filtered_docs_dict["context_input_ids"].shape)
        #print("context_attention_mask", filtered_docs_dict["context_attention_mask"].shape)
        #print("doc_scores", doc_scores.shape)
        #print("n_docs", self.n_docs)
        outputs = self.model.generate(
            context_input_ids=filtered_docs_dict["context_input_ids"],
            context_attention_mask=filtered_docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            n_docs = self.n_docs,
            **gen_kwargs,
        )
        return outputs, doc_ids