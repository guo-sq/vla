"""Load taxonomy.json and expose lookup structures for matching."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path


@dataclass(frozen=True)
class TaxonomyOption:
    id: str
    label: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Taxonomy:
    schema_version: str
    robocoin_reference: str
    structured_field_aliases: dict[str, tuple[str, ...]]
    atomic_actions: tuple[TaxonomyOption, ...]
    object_categories: tuple[TaxonomyOption, ...]
    scenes: tuple[TaxonomyOption, ...]

    @cached_property
    def by_dimension(self) -> dict[str, tuple[TaxonomyOption, ...]]:
        return {
            "atomic_actions": self.atomic_actions,
            "object_categories": self.object_categories,
            "scenes": self.scenes,
        }

    @cached_property
    def synonym_to_ids(self) -> dict[str, dict[str, str]]:
        """Lowercased synonym -> option id per dimension name."""
        out: dict[str, dict[str, str]] = {
            "atomic_actions": {},
            "object_categories": {},
            "scenes": {},
        }
        for dim, opts in self.by_dimension.items():
            for opt in opts:
                out[dim][opt.id.lower()] = opt.id
                for s in opt.synonyms:
                    out[dim][s.lower()] = opt.id
        return out


def default_taxonomy_path() -> Path:
    return Path(__file__).resolve().parent / "taxonomy.json"


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    p = path or default_taxonomy_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    aliases = raw["structured_field_aliases"]
    structured = {k: tuple(v) for k, v in aliases.items()}

    def parse_opts(key: str) -> tuple[TaxonomyOption, ...]:
        return tuple(
            TaxonomyOption(
                id=o["id"],
                label=o["label"],
                synonyms=tuple(o.get("synonyms", [])),
            )
            for o in raw[key]
        )

    return Taxonomy(
        schema_version=raw["schema_version"],
        robocoin_reference=raw["robocoin_datamanager_reference"],
        structured_field_aliases=structured,
        atomic_actions=parse_opts("atomic_actions"),
        object_categories=parse_opts("object_categories"),
        scenes=parse_opts("scenes"),
    )
