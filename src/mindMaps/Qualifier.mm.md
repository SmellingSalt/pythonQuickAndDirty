---
title: markmap
markmap:
  colorFreezeLevel: 2
---
## Situation [2]
- What is NAS
  - Goal of NAS
  - How does NAS work?
    - Different styles of NAS Algos
      - BO
        - Pro
        - Con
      - ZC
        - Pro
        - Con
      - OneShot
        - Pro
        - Con
    - Commonality
      - Search Phase
      - Final Deployment Phase
      - Expectation
        - Final model generalises to target task
          - Within
          - Across
## Complication/Gap 1 [3]
- There is a need to run NAS on new dataset
  - Ranking of NAS architectures do not hold across tasks
    - NAS360
  - Most NAS algorithms are validated and developed on image tasks only
    - NAS360
- NAS is computationally Expensive
- Have to find good generalising architecture.
- Hurdles to Generalisation
  - A Review on Generalisation in Neural Architecture Search
    - Generalisation Problems
      - What?
        - Rank Disorder
        - Meta Overfitting within Task
        - Meta Overfitting across Tasks
      - Why care?
        - Wasted resources searching
    - Gaps
      - Gap 1
        - Situation
          - Given a new dataset
          - History of past datasets and NAS search information
          - How to find NAS that will find architecture that generalises well?
        - Complication
          - Wall clock time to search
            - No direct relation with Generalisation ability of final architecture
          - Final Architecture performance
            - Generalisation gap of final architecture can be estimated
          - No information about how the NAS search did.
            - Was the architecture found lucky?
              - Inconsistent architectures on consecutive runs
            - Wastage of resources
              - Did the search focus on source tasks only?
                - Meta overfitting to sources or within task
              - Did it overfit to the validation set?
                - Meta overfitting within task
              - Did it find a configuration but throw it away?
                - Overtuning
## Research Question [3]
- To what extent does incorporating generalisation quantities improve the prediction of selecting the right NAS Algorithm that will find generalisable architectures on a new dataset?
- To what extent can dataset grouping cluster datasets to according to the above?
- To what extent can dataset grouping and generaisation quantities be used to suggest NAS algorithm on a new dataset?
- Answer
  - Clustering Image
  - Explore dataset representations.
    - Start with TransNAS style, since it is known to improve NAS algorithms
## Complication/Gap 2.1
- Joint NAS + HPO claims to be faster and produce better models: Holy grail
- To what extent can my quantities be used to measure the generalisation improvements?
  - If it claims to improve generalisation, test it on an unreliable and fast NAS.
## Complications/Gap 2.2
- Dataset awareness improved generalisation of BO.
- To what extent can a dataset awareness that is correlated with the generalisation of NAS algorithms be used to improve generalisation of unreliable and fast NAS?
  - Answer: Use the similarity measure develeoped to pass information about dataset to Zero Cost Proxies.
## Complication/Gap 3
- Dataset similarity has been shown to imporve Bayesian Optimisation NAS.
- How much of an improvement using the data similarity I proposed can be observed in the quantities I proposed?
  - Answer: Develop a Bayesian dataset similarity NAS algorithm that incorporates the chosen dataset simlarity metric in its search.

## Main Chapters of Thesis [1]
- Quantifying Generalisation in NAS
- Effect of Joint HPO and NAS on generalisation ability of NAS method(s)
- Effect of Dataset awareness on generalisation ability of NAS method(s)

## Conclusion [1]
- 