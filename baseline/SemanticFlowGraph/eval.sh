python inference.py \
   --gpu 0 \
   --data-dpath "/data0/xiaoyez/CodeContextModel/data/mylyn" \
   --config "BERTOverflow" \
   --embeddings_comparison "average" \
   --checkpoint "./data/"$name"/model_SemanticCodebert_"$name"_RN_bertoverflow_QD_q256_d256_dim128_cosine_hunks"

