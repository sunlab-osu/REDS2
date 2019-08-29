# REDS2: Relation Extraction with 2-hop DS.
Code and dataset for our paper "Leveraging 2-hop Distant Supervision from Table Entity Pairs for Relation Extraction" (EMNLP'19). Please kindly cite the following paper if you find this repo useful.
## Dataset
### Source Data
* NYT10: originally released by [Riedel et al. (2010)](http://iesl.cs.umass.edu/riedel/ecml/). We use the processed version from [OpenNRE](https://github.com/thunlp/OpenNRE).
* WikiTable: released by [Bha-gavatula et al. (2015)](http://websail-fe.cs.northwestern.edu/TabEL/#content-code)
### Data Format
We follow the data format of [OpenNRE](https://github.com/thunlp/OpenNRE), with minor modification for training/testing data under our setting.
#### Training & Testing Data
```
[
    {
        'sentence': 'Bill Gates is the founder of Microsoft .',
        'head': {'word': 'Bill Gates', 'id': 'm.03_3d', ...(other information)},
        'tail': {'word': 'Microsoft', 'id': 'm.07dfk', ...(other information)},
        'relation': 'founder',
        'is_extend': 1
    },
    ...
]
```
#### Word Embedding Data
```
[
    {'word': 'the', 'vec': [0.418, 0.24968, ...]},
    {'word': ',', 'vec': [0.013441, 0.23682, ...]},
    ...
]
```
#### Relation Mapping Data
```
{
    'NA': 0,
    'relation_1': 1,
    'relation_2': 2,
    ...
}
```
You can download the processed data used in the paper from [Box](https://osu.box.com/v/REDS2-EMNLP19)
## Software
This codebase is developped based on [pytorch-template](https://github.com/victoresque/pytorch-template), and adaptes some implementation from [OpenNRE](https://github.com/thunlp/OpenNRE).
### Requirements
* Python 3.6
* PyTorch 1.1.0

You can also use the docker image from [DockerHub](https://hub.docker.com/r/xdeng/pytorch)
### Installation and Quick Start
1. Clone the repository
2. Install all the required package or use the docker image above
3. Prepare data, config, trained model. The configs used in the paper is already included. You can get the trained model from [Box](). The final structure should look like this:
```
REDS2
|-- ... 
|-- data
|   |-- {DATASET_NAME_1}
|       |-- train.json
|       |-- test.json
|       |-- word_vec.json
|       |-- rel2id.json
|
|-- config
|   |-- config_name.conf
|
|-- saved
    |-- models
        |-- {MODEL_NAME_1}
            |-- {TRAINING_ID}
                |-- model.pth
                |-- config.json
```
4. run the command bellow to train a BASE model from scratch
```
python train.py -d {GPU_ID} -c {PATH_TO_CONFIG}
```
5. run the command bellow to train REDS2 based on pretrained BASE model
```
python train_finetune.py -d {GPU_ID} -c {PATH_TO_CONFIG} -p {PATH_TO_TRAINED_BASE_MODEL}
```
6. run the commad bellow for evaluation. If you want to test BASE+MERGE, choose trained BASE model, then pass `-m 1` to `test.py`. This will merge 2-hop sentences. 
```
python test.py -r {PATH_TO_MODEL}
```
### Add New Models
All the models are defined in `model/model.py`. There are several sub-modules defined in `model/embedding.py`, `model/encoder.py` and `model/selector.py`. You can use them or add your own.
### Configs
The config file is a json file which contain all the parameters used in the experiment. The configs used in the paper are included in the repository.
### Data Loader
Data loader is defined in `data_loader/data_loaders.py`. Use the `method` argument to choose 1-hop sentence bag, merged 1-hop & 2-hop or seperated 1-hop & 2-hop.

 





