# GNN-CONTEXT: A GNN-based Code Context Prediction Approach for Software Development Task

### Requirements
```
Python 3.9+
pip install -r requirements.txt # install the packages
```

### Reproduce the results
1. Dataset formation (`dataset_formation/`)
    - run the `run.sh` in the directory to generate the expanded code context model and variable collapsed code context model for 1-step/2-step/3-step.
        - `--input_dir`: dataset path
        - `--action [generate | display | clear]` 
            - generate: generate the graph for d-step 
            - display: show the generated graph in the input_dir
            - clear: delete all generated files

2. GNN model training (`code_context_model/`)
    - Use following command to embed the graph nodes on different embedding models
    ```
    python code_context_model/embedding.py --input_dir <dataset_dir> --output_dir <code_embedding_dir>
    ```
    - Use following command to generate the train, validation, and test dataset
    ```
    python code_context_model/build_dataset.py --input_dir <dataset_dir> --embedding_dir <code_embedding_dir> --output_dir <out_dir> --step <step> --embedding_model <emebdding_model>
    ```
    - Run the shell script `run.sh` to train the model, please set the arguments in the script

3. Baseline (`baseline`) 
    We implemented SOTA and node embedding baselines in this directory.

4. Analyis (`analyis`)
    Code for Discussion section.