# EMKD
Implementation of the paper "Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation".

## Run beauty
```
python main.py --template train_bert --dataset_code beauty
```

## Run toys
```
python main.py --template train_bert --dataset_code toys
```

## Run ml-1m
```
python main.py --template train_bert --dataset_code ml-1m
```

## Acknowledgements
Training pipeline is implemented based on this repo https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch . We would like to thank the contributors for their work.

## Citation
Please cite our paper if you find our codes useful:

```
@inproceedings{EMKD,
  author    = {Hanwen Du and
               Huanhuan Yuan and
               Pengpeng Zhao and
               Fuzhen Zhuang and
               Guanfeng Liu and
               Lei Zhao and
               Yanchi Liu and
               Victor S. Sheng},
  title     = {Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation},
  booktitle = {SIGIR},
  year      = {2023}
}

