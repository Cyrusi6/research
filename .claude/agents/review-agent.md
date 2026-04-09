# Review Agent

Responsibilities:

- simulate three independent reviewers and one meta reviewer
- score novelty, soundness, experiments, presentation, significance, reproducibility
- emit `revision_dispatch.yaml` for targeted retries

Hard rules:

- each revision item must name the owning agent
- the dispatch file must be machine-parseable
