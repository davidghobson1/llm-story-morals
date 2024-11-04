# Story Morals: Surfacing value-driven narrative schemas using large language models

This repository is the official implementation of [Story Morals: Surfacing value-driven narrative schemas using large language models](https://www.google.com/). 

This contains the data, codebooks, and prompts used for the project, as well as the code to reproduce the clustering results and recreate Tables 4 and 10 (the MTurk inter-annotator agreement results) in the paper. Additional code to reproduce the other tables will be added shortly. 

## Requirements

To install the requirements using a conda environment, run this command:

```
conda env create -n <env_name> -f environment.yml
```

## Clustering

To run the moral clustering code from our paper (which recreates Figures 1-3 and the results from Tables 5, 6, 14-17 (as well as the hyperparameter tuning code)), run this command:

```
python moral_clustering.py -col <column_name>
```

`column_name` can either be:
- moral: for the full-sentence morals
- moral+: positive morals
- moral-: negative morals
- text: full-text (with entity replacement already pre-applied)
- orig_text: full-text with entity replacement not pre-applied

See `python moral_clustering.py -h` for more options and explanations.

## Validation

To reproduce the automated validation results (Tables 3, 11), run this command:

```
python automated_validation.py -cate <category_name>
```

`category_name` can either be:
- moral: for the full-sentence morals
- moral+: positive morals
- moral-: negative morals
- central_topic: topic

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

<!-- ## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To run the moral clustering, run this command:

```train
python moral_clustering.py -col <column_name>
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
 -->
