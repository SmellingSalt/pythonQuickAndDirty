---
title: markmap
markmap:
  colorFreezeLevel: 2
---
## Situation
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

## Complication/Gap
- There is a need to run NAS on new dataset
  - Ranking of NAS architectures do not hold across tasks
    - NAS360
  - Most NAS algorithms are validated and developed on image tasks only
    - NAS360
- NAS is computationally Expensive
- Is the search worthwhile?
  - Generalisation issues are different for different NAS algorithms
    - A Review on Generalisation in Neural Architecture Search
      - Generalisation Problems
        - Rank Disorder
          - What
          - Why care
        - Meta Overfitting within Task
          - What
          - Why care
        - Meta Overfitting across Tasks
          - What
          - Why care
    - Gaps
      - Gap 1
        - Situation
          - Given a new dataset
          - History of pas datasets and NAS search information
          - Want to suggest which NAS algorithms would be worthwhile, given computational resources
        - Problems
          - How to NAS1 performance on dataset D1
            - Average final accuracy of final architectures found on "similar" datasets can vary
            - Running So many not feasible. 15k architectures, 2 mins per architecture alone takes 20 days
          - Methodology
            - Observe patterns of MOE and OT across dataset/algo pairs.
            - Datasets cluster?
            - Combination of both better than using only 1?
          - How to predict NAS computational expense for dataset D1
            - Wall clock time estimate for D1 on system with resources R1
## Research Question

## Methodology

## Chapters of Thesis

## Conclusion