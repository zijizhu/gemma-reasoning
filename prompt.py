prompt = ("The following is a first-order logic (FOL) problem.\n"
          "The problem is to determine whether the conclusion follows from the premises.\n"
          "The premises are given in the form of a set of first-order logic sentences.\n"
          "The conclusion is given in the form of a single first-order logic sentence.\n"
          "The task is to translate each of the premises and conclusions into FOL expressions," 
          "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
          "Expressions should be adhere to the format of the Python NLTK package logic module.\n\n"
          "<PREMISES>\n"
          "{premises}\n"
          "<PREMISES\n>")



def get_instructions(self):
        instructions = ""
        instructions += "The following is a first-order logic (FOL) problem.\n"
        instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
        instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
        instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
        if self._mode == "baseline":
            instructions += f"The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
        else:
            instructions += "The task is to translate each of the premises and conclusions into FOL expressions, "
            if self._mode == "scratchpad":
                instructions += f"and then to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
            elif self._mode == "neurosymbolic":
                instructions += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
                instructions += "Expressions should be adhere to the format of the Python NLTK package logic module."
        return instructions + "\n\n"