python model/inference.py \
   --gpu 0 \
   --step 1 \
   --data-dpath "/data0/xiaoyez/CodeContextModel/data/mylyn" \
   --embeddings-comparison "average" \
   --checkpoint "/data0/xiaoyez/CodeContextModel/baseline/SemanticFlowGraph/data/model_SemanticCodebert_mylyn_bertoverflow_CodeContext_q256_d256_dim128_cosine" \
   --config "BERTOverflow" \
   --amp


