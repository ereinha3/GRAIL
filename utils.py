def get_max_prompt_length(dataset, tokenizer, prefix_count):
    max_length = 0
    for problem in dataset:
        problem_with_cue = add_solution_to_prompt(problem['problem_text'], problem['path'])
        embeddings = tokenizer.encode(problem_with_cue)
        max_length = max(max_length, len(embeddings) + prefix_count)
            
    return max_length

def add_solution_to_prompt(prompt, solution):
    solution = [str(s) for s in solution]
    sol_str = "[" + ", ".join(solution) + "]"
    full_text = prompt + '\nSolution: ' + sol_str
    return full_text

def add_solution_cue_to_prompt(prompt):
    return prompt + '\nSolution: '