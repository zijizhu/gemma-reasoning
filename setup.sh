#! /bin/bash

set -x

python -m pip install spacy lightning transformers datasets scikit-learn tqdm

wget "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
unzip "proofwriter-dataset-V2020.12.3"