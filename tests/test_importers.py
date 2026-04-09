import json
from pathlib import Path

from auto_research.importers import ConsensusImporter, parse_ris
from auto_research.itr_ideas import build_itr_theme_map
from auto_research.workspace import init_workspace


def test_consensus_importer_registers_raw_normalized_and_extracted(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
    }
    paths = init_workspace(config, "consensus topic", project_id="proj_consensus", simulate=True)
    source = tmp_path / "consensus.json"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Query: image-text retrieval hard negatives"},
                    {"role": "assistant", "content": 'Consider "FiCo-ITR" and https://arxiv.org/abs/2407.20114'},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ConsensusImporter(paths.root).import_file(source, label="demo")

    assert result["raw"].endswith("demo.raw.txt")
    extracted = json.loads((paths.root / "literature" / "imports" / "consensus" / "demo.extracted.json").read_text(encoding="utf-8"))
    assert "image-text retrieval hard negatives" in extracted["queries"]
    assert any("2407.20114" in url for url in extracted["urls"])
    assert "FiCo-ITR" in extracted["paper_title_candidates"]


def test_consensus_importer_supports_ris_and_theme_mapping(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
    }
    paths = init_workspace(config, "ris topic", project_id="proj_ris", simulate=True)
    source = tmp_path / "consensus.ris"
    source.write_text(
        "\n".join(
            [
                "TY  - JOUR",
                "TI  - Modality-specific adaptive scaling and attention network for cross-modal retrieval",
                "PY  - 2024",
                "DO  - 10.1016/j.neucom.2024.128664",
                "UR  - https://consensus.app/papers/example",
                "ER  - ",
                "TY  - JOUR",
                "TI  - Multimodal Alignment and Fusion: A Survey",
                "PY  - 2024",
                "ER  - ",
            ]
        ),
        encoding="utf-8",
    )

    result = ConsensusImporter(paths.root).import_file(source, label="ris_demo")
    extracted = json.loads((paths.root / "literature" / "imports" / "consensus" / "ris_demo.extracted.json").read_text(encoding="utf-8"))
    theme_map = build_itr_theme_map(extracted["entries"])

    assert result["raw"].endswith("ris_demo.raw.txt")
    assert len(parse_ris(source.read_text(encoding="utf-8"))) == 2
    assert extracted["import_type"] == "consensus_ris"
    assert any("cross-modal retrieval" in query.lower() for query in extracted["queries"])
    assert theme_map["direct_retrieval"]
    assert theme_map["themes"]["adaptive_dynamic_attention"]
