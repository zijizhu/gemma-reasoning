import json
import torch
import argparse
from fol import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from folio_dataset import create_prompt, instruction, get_example_prompt_str, fol_to_nltk


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Logical Inference with FOL Translation')
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, choices=['FOLIO', 'PROOFWRITER'])
    args = parser.parse_args()

    if args.dataset == 'FOLIO':
        dataset_train = load_dataset('yale-nlp/FOLIO', split='train')
        dataset_val = load_dataset('yale-nlp/FOLIO', split='validation')
    else:
        raise NotImplementedError

    dataset_train = dataset_train.map(fol_to_nltk)
    dataset_val = dataset_val.map(fol_to_nltk)

    example_prompt_str = get_example_prompt_str(dataset_train, n_shots=4)

    dataset_val = dataset_val.map(
        create_prompt,
        fn_kwargs=dict(
            instruction=instruction,
            example_prompt_str=example_prompt_str
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


    results = []
    for example in dataset_val:
        input_ids = tokenizer(example['prompt'], return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_length=1024)
        output_text = tokenizer._decode(outputs)[0]
        print(output_text)
        print()

        llm_translated_fols = []
        for line in output_text.splitlines():
            if not line.startswith('FOL'):
                continue
            llm_translated_fols.append(line[5:])

        res = evaluate(llm_translated_fols[:-1], llm_translated_fols[-1])
        results.append(res)
    
    with open(args.outdir, 'w+') as fp:
        json.dump(results, f'results/{args.model.split("/")[-1]}_{args.dataset}_results.json')
