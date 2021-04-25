##  Deep Learning Solutions for Question Answering with Computation Efficiency and Interpretability
### Purpose
This project contains all the source code for the final paper *Experimenting SQuAD Problem with Enhanced Data Efficiency,Computation Efficiency and Interpretability*.


### Usage

#### Enviroment Configuration

```
> pip install -r requirements.txt
```
#### Preprocess Data and EDA
The Data Preprocessing and EDA process is implemented in Jupyter Notebook for clear understading.
```
> cd Final_Code/Preprocessing
> jupyter notebook
```
#### Model
BiDAF
```
### Enviromen Setup ###
> python ./Model/BiDAF/setup.py
### Model Training ###
> python ./Model/BiDAF/train.py -n baseline
### Model Prediction ###
> python ./Model/BiDAF/test.py -n baseline __model_path <model_save_dir>
```
Bert
```
### Model Training ###
python run_squad.py \
    --model_type bert \
    --model_name_or_path <bert_save_file> \
    --do_train \
    --do_lower_case \
    --train_file <train_file_dir> \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 768 \
    --doc_stride 128 \
    --output_dir <model_output_dir>
### Model Prediction ###
python run_squad.py \
    --model_name_or_path <bert_save_file> \
    --do_eval \
    --predict_file <dev_file_dir> \
    --output_dir <result_output_dir> 
```



#### Evaluation
Calculate EM and F1 score for each model.
```
> python ./Evaluation/eval.py <result.json> <prediction.json>
```

#### Interpretability
Visualize Layer-wise Transformer representation and Token level Attribution Score 
```
>python ./Interpretability/hidden_state_visualize.py <context_answer_pairs.json>
```
### Contributors
* [Lu Guo](luguo@u.nus.edu)
* [Qing Han](qing.h@u.nus.edu)
* [Wenqian Li](wenqian@u.nus.edu)
* [Wenxi Yu](wenxi.yu@u.nus.edu)
* [Junzhao Yu](junzhao.yu@u.nus.edu)


### Project structure
```
Final_Code
├─ Intepretability
│  ├─ data_utils.py
│  ├─ Fresno.json
│  ├─ hidden_state_visualizer.py
│  ├─ model_wrapper.py
│  ├─ plotting.py
│  ├─ requirements.txt
│  ├─ sample.json
│  └─ sample_wrapper.py
├─ LICENSE
├─ Model
│  ├─ Bert
│  │  ├─ api.py
│  │  ├─ bert.py
│  │  ├─ dev-v2.0.json
│  │  ├─ dev.json
│  │  ├─ img
│  │  │  └─ postman.png
│  │  ├─ LICENSE
│  │  ├─ requirements.txt
│  │  ├─ test
│  │  ├─ train-v2.0.json
│  │  ├─ training
│  │  │  ├─ run_squad.py
│  │  │  ├─ utils_squad.py
│  │  │  ├─ utils_squad_evaluate.py
│  │  ├─ utils.py
│  └─ BiDAF
│     ├─ args.py
│     ├─ command.txt
│     ├─ data
│     ├─ environment.yml
│     ├─ layers.py
│     ├─ LICENSE
│     ├─ models.py
│     ├─ save
│     ├─ setup.py
│     ├─ test.py
│     ├─ train.py
│     └─ util.py
├─ Preprocessing
│  └─ Preprocessing.ipynb
├─ Evaluation
│  ├─ bert_10.csv
│  ├─ bert_100.csv
│  ├─ bert_100_predictions_.json
│  ├─ bert_10_predictions_.json
│  ├─ bert_large.csv
│  ├─ bert_large_predictions.json
│  ├─ dev-v2.0.json
│  ├─ dev-v2.0_q_type.csv
│  ├─ dev.json
│  ├─ dev_q_type.csv
│  ├─ eval.py
│  └─ score_by_model_type.xlsx
└─ README.md

```

### License

[MIT](LICENSE) © Richard Littauer
