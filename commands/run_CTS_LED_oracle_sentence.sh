python CTS_LED_trainer.py --train_retrieved_ids sorted_sentence_ROUGE_train.jsonl --distant_retrieved_ids sorted_sentence_ROUGE_distant.jsonl --dev_retrieved_ids sorted_sentence_ROUGE_test.jsonl --dominant_only --auto_regressive --checkpoint oracle_sentence_CTS_citation_span_generator_final --epoch 5 --passages_path cited_text_embeddings_sentence_better/cited_papers --n_docs 10 --context_window_size 2 --max_input_length 2048