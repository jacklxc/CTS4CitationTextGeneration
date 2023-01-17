python train_RAG.py --per_device_train_batch_size 3 --per_device_eval_batch_size 1 --dominant_only --n_docs 5 --checkpoint_dir FiD_CTS_RAG_span_generator_final.model --distant_dataset /home/data/XiangciLi/CORWA/CORWA_distant --train_dataset /home/data/XiangciLi/CORWA/annotated_train --train_oracle_ids sorted_ROUGE_train.jsonl --output_dir FiD_CTS_RAG_span_generation_final.jsonl --use_fid --num_train_epochs 5 --generator_path bart_base_span_generator_cdlm/checkpoint-2000 --auto_regressive --max_combined_length 650 --learning_rate 5e-5