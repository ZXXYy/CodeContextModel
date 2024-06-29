# code_context_model/run.sh
python train.py --input_dir "" --do_train --do_test --output_dir "" --num_epochs 50 --lr 1e-4 --threshold 0.5 --seed 42
