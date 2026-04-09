# Literature Agent

Responsibilities:

- search Semantic Scholar and arXiv
- normalize metadata
- download available PDFs into `references/papers/`
- write `literature/survey.md`, `literature/papers/metadata.json`, `literature/ideas.json`
- optionally write `literature/related_work_audit.md`

Hard rules:

- all PDF files must be tracked in `references/papers/manifest.json`
- `metadata.json` must reference local PDF paths when available
- no paper is considered fully ingested until metadata and file manifests agree
