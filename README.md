# Story Morals: Surfacing value-driven narrative schemas using large language models

This repository is the official implementation of [Story Morals: Surfacing value-driven narrative schemas using large language models](https://aclanthology.org/2024.emnlp-main.723/). 

This contains the data, codebooks, and prompts used for the project, as well as the code to reproduce our results. Below is the breakdown of the repo structure.

<p align="center">
    <img src="https://github.com/davidghobson1/llm-story-morals/blob/main/images/moral_pipeline.png?raw=true" alt="Results Image" width="550"/>
</p>

## Repo Structure
```
.
├── code                                             
    ├── Categorical-Automated-Validation.ipynb       # automated comparison of protagonist/antagonist/valence/protagonist type
    ├── MTurk-Analysis.ipynb                         # analyse MTurk results
    ├── automated_validation.py                      # automated comparison of human vs GPT morals
    ├── moral_clustering.py                          # cluster the morals 
|
├── codebooks                                        
    ├── mturk_survey_template.html                   # Amazon Mechanical Turk validation template/instructions
    ├── story_morals_codebook.docx                   # codebook for human story moral annotations
|
├── data                                              
    ├── application                                  # data for Application section of paper
        ├── fairytalez_dataset.csv
        ├── gpt_responses_fairytalex.csv
    ├── validation                                   # data for Validation section of paper
        ├── moral_annotations                        
            ├── gpt_responses_english.csv            # GPT's responses to the English validation texts
            ├── gpt_responses_mandarin.csv           # GPT's responses to the Mandarin validation texts
            ├── human_responses_english.csv          # human responses to the English validation texts
            ├── human_responses_mandarin.csv         # human responses to the Mandarin validation texts
        ├── mturk
            ├── mturk_responses.csv                  # responses from MTurk survey
        ├── validation_dataset.csv                   # text and metadata for the validation dataset 
|
├── prompts                                            
    ├── prompts.json                                 # prompts used in our work
|
├── requirements.txt                                  
├── README.md
```

## Requirements

To install the requirements using a Python virtual environment, use the requirements.txt file:

```
python -m venv <myenv_name>
source <myenv_name>/bin/activate
python -m pip install -r requirements.txt
```

Unfortunately, I haven't been able to reliably create a conda environment that always runs without errors. I'll include an environment.yml file when (and if) I do. Apologies for any inconvenience in the meantime.

## Running the Code

### Notebooks

The two notebooks include the automated comparison between human and GPT responses for the protagonist, antagonist, valence, and protagonist type questions, and the inter-annotator agreement from our MTurk survery.

Both notebooks can be run as-is and use the data existing in this repo.

### Clustering

To run the moral clustering results, run the `moral_clustering.py` script as follows. (This recreates Figures 1-3 and the results from Tables 5, 6, 14-17, as well as the hyperparameter tuning).

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

<p align="center">
    <img src="https://github.com/davidghobson1/llm-story-morals/blob/main/images/sample_moral_clustering.png?raw=true" alt="Results Image" width="400"/>
</p>

### Validation

To run the automated comparison between human and GPT responses for the moral, moral+, moral-, and central topic questions, run the `automated_validation.py` as follows. (This reproduces Tables 3 and 11).

```
python automated_validation.py -cate <category_name>
```

`category_name` can either be:
- moral: for the full-sentence morals
- moral+: positive morals
- moral-: negative morals
- central_topic: topic

Note that this script may take a while to run (~30 mins) if running on CPU because BERTScore can be quite slow.

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

<!-- ## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
 -->
