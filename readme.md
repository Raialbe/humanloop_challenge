# HumanLoop Data Challenge
## Alberto Raimondi

The task was completed using a jupyter notebook to allow the exploration of the data while explaining the motivation behind every step.

## Installation
The file `requirements.txt` contains the python libraries needed to run the model.
Additionally the spacy trf model is needed for dependency parsing and can be installed by running `python -m spacy download en_core_web_trf`

## Preprocessing
The preprocessing step required accessing the API many times. To avoid too many useless requests the result was saved in a pickle file in the repository.

## Usage
The analysis can be run by running the `main` jupyter notebook in a jupyter kernel.