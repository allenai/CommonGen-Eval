# Data

Given several concepts (i.e., nouns or verbs), we ask models to write a short and simple sentence that contains *all* the required words. 
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

Concepts: "{$concept_list}"

Model A: "{$candidate_A}"

Model B: "{$candidate_B}"

# Your Task

Your task is to choose a better sentence from the two candidates. Decide which model's sentence is better in terms of the naturalness and commonness of the scenes they describe. 

## Rules: 
- A better sentence should describe a common scene in daily life, and all concepts should be used in a natural way.
- You should prefer sentences that use all given concepts with correct part-of-speech tags. 
- A simpler and shorter sentence is preferred if it describes the same scene as the other sentence.
- If you think both sentences are equally good or bad, please choose "tie".

Now, please output your choice ("A" or "B" or "tie").

Your choice: 