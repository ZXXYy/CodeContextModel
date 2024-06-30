# code_context_model/run.sh
python code_context_model/train.py \
--do_train \
--device 9 \
--train_batch_size 1 \
--valid_batch_size 1 \
--do_test \
--test_batch_size 1 \
--input_dir "/data0/xiaoyez/CodeContextModel/data/repo_first_3" \
--embedding_dir "/data0/xiaoyez/CodeContextModel/codebert_embedding_results" \
--output_dir "/data0/xiaoyez/CodeContextModel/model_output" \
--num_epochs 50 \
--lr 1e-3 \
--threshold 0.5 \
--seed 42 \
# --debug

# debug
# python code_context_model/train.py \
# --do_train \
# --debug \
# --device 9 \
# --train_batch_size 8 \
# --valid_batch_size 1 \
# --do_test \
# --test_batch_size 1 \
# --input_dir "/data0/xiaoyez/CodeContextModel/data/repo_first_3" \
# --embedding_dir "/data0/xiaoyez/CodeContextModel/bge_embedding_results" \
# --output_dir "/data0/xiaoyez/CodeContextModel/model_output" \
# --test_model_pth "/data0/xiaoyez/CodeContextModel/model_output/model_49.pth" \
# --num_epochs 50 \
# --lr 1e-4 \
# --threshold 0.5 \
# --seed 42

