Here is a sample config file
```
{
    "name": "BasePCNNAtt", # This is the config name. All experiments using this config will be stored under /saved/models/BasePCNNAtt
    "n_gpu": 1,

    "seed": 1,
    
    "arch": {
        "type": "BasePCNNAttModel", # model name, this should match the class name in model/model.py
        "args": {
            "word_embedding_dim": 50,
            "pos_embedding_dim": 5,
            "max_length": 120,
            "hidden_size": 230,
            "kernel_size": 3,
            "stride_size": 1,
            "activation": "relu",
            "dropout_prob": 0.5
        }
    },
    "data_dir": "data/camera_ready",
    "data_loader": {
        "type": "BaseNytLoader", # this should match the data_loader name in data_loader/data_loaders.py
        "args":{
            "data_dir": "data/camera_ready",
            "max_length": 120,
            "batch_size": 250,
            "method": 0,
            "num_workers": 1
        }
    },
    "label_reweight": 0.05,
    "optimizer": {
        "type": "SGD",
        "args":{
            "lr": 0.005,
            "weight_decay": 0,
            "momentum": 0.9
        }
    },
    "loss": "cross_entropy",
    "train_metrics": [
        "accuracy", "non_na_accuracy"
    ],
    "eval_metrics": [
        "accuracy", "non_na_accuracy", "auc"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 40,
            "gamma": 0.5
        }
    },
    "trainer": {
        "epochs": 500,
        "save_dir": "saved/",
        "save_period": 4,
        "verbosity": 2,
        
        "monitor": "max val_auc",
        "early_stop": 10,
        
        "tensorboardX": true,
        "log_dir": "saved/runs"
    }
}
``` 