import random
from fol import convert_to_nltk_rep

instruction = ("The following is a first-order logic (FOL) problem.\n"
               "The problem is to determine whether the conclusion follows from the premises.\n"
               "The premises are given in the form of a set of first-order logic sentences.\n"
               "The conclusion is given in the form of a single first-order logic sentence.\n"
               "The task is to translate each of the premises and conclusions into FOL expressions," 
               "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
               "Expressions should be adhere to the format of the provided examples.\n\n")


def fol_to_nltk(example):
    example['premises-FOL-nltk'] = convert_to_nltk_rep(example['premises-FOL'])
    example['conclusion-FOL-nltk'] = convert_to_nltk_rep(example['conclusion-FOL'])
    return example


def get_example_prompt_str(dataset, n_shots=8):
    example_prompts = []
    example_columns = ['premises', 'premises-FOL-nltk', 'conclusion', 'conclusion-FOL-nltk']
    sampled_idxs = random.sample(range(len(dataset)), n_shots)
    for example in dataset.select(sampled_idxs).select_columns(example_columns).iter(batch_size=1):
        p, p_fol, c, c_fol = tuple(example[k][0] for k in example_columns)
        fol_lines_p, fol_lines_c = [], []
        for p_line, p_fol_line in zip(p.splitlines(), p_fol.splitlines()):
            fol_lines_p += [f'TEXT: {p_line}', f'FOL: {p_fol_line}']
        for c_line, c_fol_line in zip(c.splitlines(), c_fol.splitlines()):
            fol_lines_c += [f'TEXT: {c_line}', f'FOL: {c_fol_line}']
        example_prompts+= ['<PREMISES>\n', p, '\n<PREMISES>\n',
                        '<CONCLUSION>\n', c, '\n<CONCLUSION>\n',
                        '<EVALUATE>\n', '\n'.join(fol_lines_p), '\n', '\n'.join(fol_lines_c), '\n<EVALUATE>\n\n']
    example_prompt_str = ''.join(example_prompts)
    return example_prompt_str


def create_prompt(example, instruction='', example_prompt_str=''):
    assert instruction and example_prompt_str
    question = ['<PREMISES>\n', example['premises'], '\n<PREMISES>\n',
                '<CONCLUSION>\n', example['conclusion'], '\n<CONCLUSION>\n', '<EVALUATE>\n']
    example['prompt'] = instruction + example_prompt_str + ''.join(question)
    return example


def creat_prompt_proofwriter(example, instruction='', example_prompt_str=''):
    assert instruction and example_prompt_str
    question = ['<PREMISES>\n', example['premises'], '\n<PREMISES>\n',
                '<CONCLUSION>\n', '\n'.join(example['conclusion']), '\n<CONCLUSION>\n', '<EVALUATE>\n']
    return instruction + example_prompt_str + ''.join(question)
