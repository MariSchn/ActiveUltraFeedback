for each completions_batch:
    tokenize, if:
        - acquisition function relies on reward model
        - features are not precomputed
        - in other words, we can skip this step because we assume precomputed features :)
    
    compute rewards, if
        - acquisition function relies on reward model
        


