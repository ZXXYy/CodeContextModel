# code_context_model/run.sh
python train.py \
--do_train \
--do_test \
--input_dir "/data0/xiaoyez/CodeContextModel/data/repo_first_3" \
--embedding_dir "/data0/xiaoyez/CodeContextModel/bge_embedding_results" \
--output_dir "/data0/xiaoyez/CodeContextModel/model_output" \
--num_epochs 50 \
--lr 1e-4 \
--threshold 0.5 \
--seed 42

