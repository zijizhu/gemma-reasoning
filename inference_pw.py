import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from fol import evaluate
from lightning import seed_everything
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from folio_dataset import instruction, get_example_prompt_str, fol_to_nltk, creat_prompt_proofwriter, convert_to_nltk_rep


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Logical Inference with FOL Translation')
    parser.add_argument('--model', type=str)
    parser.add_argument('--depth', type=int)
    args = parser.parse_args()

    seed_everything(42)
    
    dataset_train = load_dataset('yale-nlp/FOLIO', split='train')
    dataset_train = dataset_train.map(fol_to_nltk)
    example_prompt_str = get_example_prompt_str(dataset_train, n_shots=4)

    proofwriter_test = pd.read_json(f'proofwriter-dataset-V2020.12.3/OWA/depth-{args.depth}/meta-test.jsonl', lines=True)

    all_examples = []
    for index, row in proofwriter_test.iterrows():
        premises = row['theory']
        premises = premises.replace('. ', '\n ')
        question_dict = row['questions']
        num_q = len(question_dict)
        questions, answers = [], []
        for i in range(1, num_q + 1):
            questions.append(question_dict[f'Q{i}']['question'])
            answers.append(question_dict[f'Q{i}']['answer'])
        all_examples.append(dict(
            premises=premises,
            conclusions=questions,
            labels=answers
        ))
    
    all_examples = all_examples[:100]
    for example in all_examples:
        example['prompt'] = creat_prompt_proofwriter(example=example,
                                                     instruction=instruction,
                                                     example_prompt_str=example_prompt_str)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
    ).to('cuda')

    results = []
    for example in tqdm(all_examples):
        input_ids = tokenizer(example['prompt'], return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=512)
        output_text = tokenizer.batch_decode(outputs)[0]
        llm_translated_fols = []

        for line in output_text.splitlines():
            if not line.startswith('FOL'):
                continue
            llm_translated_fols.append(line[5:])

        res = evaluate([convert_to_nltk_rep(s) for s in llm_translated_fols[:-1]],
                       convert_to_nltk_rep(llm_translated_fols[-1]))
        res = evaluate(llm_translated_fols[:-1], llm_translated_fols[-1])
        results.append(res)
    
        with open(f'{args.model.split("/")[-1]}_depth{args.depth}_results.json', 'w+') as fp:
            json.dump(results, fp=fp)