python run_DPR_CTS.py --dominant_only --n_docs 20 --train_dataset /home/data/XiangciLi/CORWA/annotated_train --auto_regressive --max_combined_length 650 --per_device_train_batch_size 4 --output_dir retrieved_train_doc_ids.jsonl 
python run_DPR_CTS.py --dominant_only --n_docs 20 --train_dataset /home/data/XiangciLi/CORWA/annotated_test_Nov15 --auto_regressive --max_combined_length 650 --per_device_train_batch_size 4 --output_dir retrieved_test_doc_ids.jsonl 
python run_DPR_CTS.py --dominant_only --n_docs 20 --distant_dataset /home/data/XiangciLi/CORWA/CORWA_distant --auto_regressive --max_combined_length 650 --per_device_train_batch_size 4 --output_dir retrieved_distant_doc_ids.jsonl 
python run_DPR_CTS.py --dominant_only --n_docs 20 --train_dataset /home/data/XiangciLi/CORWA/annotated_train --auto_regressive --max_combined_length 650 --per_device_train_batch_size 4 --output_dir abstract_train_doc_ids.jsonl --cited_abstract_only
python run_DPR_CTS.py --dominant_only --n_docs 20 --train_dataset /home/data/XiangciLi/CORWA/annotated_test_Nov15 --auto_regressive --max_combined_length 650 --per_device_train_batch_size 4 --output_dir abstract_test_doc_ids.jsonl --cited_abstract_only
python run_DPR_CTS.py --dominant_only --n_docs 20 --distant_dataset /home/data/XiangciLi/CORWA/CORWA_distant --auto_regressive --max_combined_length 650 --per_device_train_batch_size 4 --output_dir abstract_distant_doc_ids.jsonl --cited_abstract_only