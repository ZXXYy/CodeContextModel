# code_context_model/run.sh
python train.py \
--train_batch_size 32 \
--valid_batch_size 1 \
--test_batch_size 1 \
--neg_sz 2 \
--pretrained_emb_path "/data0/xiaoyez/CodeContextModel/baseline/GNNAPIRec/word2vec.pretrain" \
--input_dirs  "/data2/xiaoyez/CodeContextModel/dataset_word2vec_step3"  \
--output_dir "/data2/xiaoyez/CodeContextModel/GNNAPIRec/model_output_step3" \
--num_epochs 50 \
--lr 1e-5 \
--seed 42 \
--device 0 \
--do_test \
--test_model_pth "/data2/xiaoyez/CodeContextModel/GNNAPIRec/model_output_step3/03-08-16-21/model_39.pth" 

# --test_model_pth "/data2/xiaoyez/CodeContextModel/GNNAPIRec/model_output_step2/03-08-16-12/model_49.pth" 
# --do_train 




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
# --threshold 0.5\ 
# --seed 42

