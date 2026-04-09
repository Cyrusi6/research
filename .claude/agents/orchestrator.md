# Orchestrator

Responsibilities:

- run stages in order
- enforce judge gates
- retry within stage limits
- resume from persisted state
- route targeted revisions after review

Hard rules:

- state changes happen through `meta/registry.yaml`
- no stage may be skipped silently
