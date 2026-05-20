# LongTail-Swap: benchmarking language models’ abilities on rare words.

This repository contains:

1) Code to create the LT-Swap tasks on a given text dataset 

2) LT-Swap10M and LT-Swap100M constructed on the 10M and 100M version of the BabyLM datasets

## Installation

Get the latest version of transformers, multiprocessing, pyspellchecker, numpy, nltk and pytorch


## Sentence generation using API 

Our pipeline relies on calling an LLM API to generate and filter sentences. You can find the Jupyter notebook for generation at generate_task/generation_notebook.ipynb. By default, the notebook requires an OpenAI API key, but the code is generic enough to allow you to integrate any API of your choice. Please refer to the instructions at the beginning of the notebook for more details.

## Creating tasks from a pretraining set

The whole process is identical for any English text dataset. As an example we set $PRETRAINING_DIR to the BabyLM 10M words text datasets (download from https://babylm.github.io/).

First step is to create a list of word candidates and word inflections for each file in the BabyLM datasets.
```
PRETRAINING_DIR='BabyLM_2024/train_10M' #path to pretraining dir, make sure it contains only text files
TASK_DIR='BabyLM_2024/task_files' #path to task directory, will be created automatically
NCPUS=5  # Number of CPU cores to use; ideally, match this to the number of text files in the pretraining directory
python generate_task/get_word_lists.py  --data=$PRETRAINING_DIR/ --output_wordlists_dir=$TASK_DIR/wordlists/ --ncpus=5
```
We merge those words lists and create two files: one list of words for WordSwap and one list of inflected pairs for InflectionSwap and AgreementSwap
```
python generate_task/build_longtail.py --wordlists_dir=$TASK_DIR/wordlists/ --output_wordlist=$TASK_DIR/longtail_wordlist --output_inflpairs=$TASK_DIR/longtail_inflpairs --output_voc=$TASK_DIR/vocabulary
```

### Creating WordSwap

This prompts will directly ask an LLM to generate sentences for WordSwap.
```
python generate_task/wordswap_sentence_prompts.py --wordlist=$TASK_DIR/longtail_wordlist --output_file=$TASK_DIR/wordswap_sentence_prompts
```
Use the notebook generation_notebook.ipynb in "generation" mode with the file "TASK_DIR/wordswap_sentence_prompts" as input. Save the output of the script at "TASK_DIR/wordswap_sentence_generations"

Then, the next script creates the prompts for the LLM filtering step. The LLM may use words not present in the pretraining set, by default such generations are filtered out, but you may edit line 46 to allow this.
```
python generate_task/wordswap_pairs_and_filtering_prompts.py --input_file=$TASK_DIR/wordswap_sentence_generations --output_file=$TASK_DIR/wordswap_sentence_pairs_filtering_prompts --voc_file=$TASK_DIR/vocabulary
```
Finally, use again the notebook generation_notebook.ipynb in the "filtering" mode and the file "TASK_DIR/wordswap_sentence_pairs_filtering_prompts" as input. The output is the final WordSwap task: a text file with the sentence pairs that passed the filter. This last step is quite long and can take last several days, up to a week depending on your API subscription plan.

### Creating InflectionSwap and AgreementSwap

For InflectionSwap and AgreementSwap we first ask an LLM if the automatically computed inflected pairs are indeed inflections of the same word. In addition for AgreementSwap we ask the LLM if the inflected pairs are words that could take a reflexive pronoun.
```
python generate_task/inflpairs_filtering_prompts.py --inflpairs=$TASK_DIR/longtail_inflpairs --output_file=$TASK_DIR/syntax_words_filtering_prompts
```
The prompts are sent to the LLM for filtering
Use the notebook generation_notebook.ipynb in "generation" mode with the file "TASK_DIR/syntax_words_filtering_prompts" as input. Save the output of the script at "TASK_DIR/syntax_words_to_be_filtered"

Then the next script filters words and create the prompts for AgreementSwap and InflectionSwap sentence generations
```
python generate_task/syntax_sentence_prompts.py --inflpairs=$TASK_DIR/syntax_words_to_be_filtered --output_file=$TASK_DIR/syntax_sentence_pairs_prompts
```
Use the notebook generation_notebook.ipynb in "generation" mode with the file "TASK_DIR/syntax_sentence_pairs_prompts" as input. Save the output of the script at "TASK_DIR/syntax_sentence_generations"

Then, the next script creates the prompts for the LLM filtering step. The LLM may use words not present in the pretraining set, by default such generations are filtered out, but you may edit line 203 to allow this.
```
python generate_task/syntax_get_generation_and_filtering_prompts.py --input_file=$TASK_DIR/syntax_sentence_generations --output_file=$TASK_DIR/syntax_sentence_pairs_filtering_prompts --voc_file=$TASK_DIR/vocabulary
```

Finally, use again the notebook generation_notebook.ipynb in the "filtering" mode and the file "TASK_DIR/syntax_sentence_pairs_filtering_prompts" as input. The output is the final Agreement and InflectionSwap tasks: a text file with the sentence pairs that passed the filter. This last step is quite long and can take last several days, up to a week depending on your API subscription plan.

The sentence pairs for agreementswap and inflectionswap are gathered in one file, in order to split them in two files use the following script.
```
python generate_task/split_agrswap_infswap.py $TASK_DIR/syntax_sentence_pairs $TASK_DIR/inflswap_sentence_pairs_10M $TASK_DIR/agrswap_sentence_pairs_10M
```

## Evaluating LM on LT-Swap10M and LT-Swap100M

LT-Swap10M and LT-Swap100M are created based on the BabyLM 10M and 100M words text datasets. Each task is composed of three subtask files for WordSwap, InflectionSwap and AgreementSwap. 

For WordSwap, each line is formatted as follows.
```
<frequency bin index> <rule> <target word 1> <pretraining sentence 1> <index of word 1 in pretraining sentence 1> <generated sentence 1> <index of word 1 in generated sentence 1> <target word 2> <pretraining sentence 2> <index of word 2 in pretraining sentence 2> <generated sentence 2> <index of word 2 in generated sentence 2>
```

The pretraining sentences serve for the prefix-method described in the paper (optional and only for WordSwap). For InflectionSwap and AgreementSwap, the format of each line is the same without the pretraining sentences.

In order to evaluate a model on one of the task use the following script.
```
TASK_TYPE='wordswap' #choose among wordswap,inflswap or agrswap
MODEL_NAME='babylm/babyllama-10m-2024' #list of allowed models in eval_longtail.py line 74
PAIR_FILE='babylm-lt-swap/wordswap_pairs_10M'
RESULT_DIR='babylm-lt-swap/results'
python eval/eval_longtail.py --task_type=$TASK_TYPE --pair_file=$PAIR_FILE --model_name=$MODEL_NAME --output_dir=$RESULT_DIR
```

And to reproduce the metrics (spread ratio, correlation between frequency bins and model performances), use the following script.
```
python eval/get_correlations.py $RESULT_DIR
```

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

LT-Swap is CC-by-NC licensed, as found in the LICENSE file.

The data is intended for benchmarking purposes and is licensed CC-by-NC. The data is an output from Llama 3.1, and subject to the [Llama 3.1 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE). Use of the data to train, fine tune, or otherwise improve an AI model, which is distributed or made available, shall also include "Llama" at the beginning of any such AI model name.


