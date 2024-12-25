python model/main.py \
   --gpu 0 \
   --n-epochs 30 \
   --dim 128 \
   --bsize 32 \
   --query_maxlen 256 \
   --doc_maxlen 256 \
   --special-tokens "CodeContext" \
   --data-dpath "/data0/xiaoyez/CodeContextModel/data/mylyn" \
   --config "BERTOverflow"