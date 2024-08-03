# code_context_model/run.sh
python code_context_model/train.py \
--model 'rgcn' \
--train_batch_size 32 \
--valid_batch_size 1 \
--test_batch_size 1 \
--input_dir  "/data0/xiaoyez/CodeContextModel/dataset_step1"  \
--embedding_dir "/data0/xiaoyez/CodeContextModel/embedding_bge" \
--output_dir "/data0/xiaoyez/CodeContextModel/model_output" \
--num_epochs 50 \
--lr 1e-6 \
--threshold 0.5 \
--pos_margin 0.8 \
--neg_margin 0.0 \
--gnn_layers 3 \
--seed 42 \
--device 9 \
--do_test \
--do_train
# --test_model_pth "/data0/xiaoyez/CodeContextModel/model_output/07-14-00-25/model_48.pth"

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

