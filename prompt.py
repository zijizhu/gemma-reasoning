prompt = ("The following is a first-order logic (FOL) problem.\n"
          "The problem is to determine whether the conclusion follows from the premises.\n"
          "The premises are given in the form of a set of first-order logic sentences.\n"
          "The conclusion is given in the form of a single first-order logic sentence.\n"
          "The task is to translate each of the premises and conclusions into FOL expressions," 
          "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
          "Expressions should be adhere to the format of the Python NLTK package logic module.\n\n")

def get_prompt(examples: list[str]):
    return prompt.format(*examples) 
