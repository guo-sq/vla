"""FastAPI server for CPT dataset selection UI."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Annotated, Any, Literal

from fastapi import Body
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field

from .indexer import ensure_repos_robot_type_column
from .matcher import QueryFilter
from .matcher import fts_search
from .matcher import query_repos
from .taxonomy_loader import load_taxonomy


class QueryBody(BaseModel):
    atomic_actions: list[str] = Field(default_factory=list)
    object_categories: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    include_incomplete_meta: bool = False
    datasets: list[str] = Field(
        default_factory=list,
        description="Dataset names (indexed family). Empty list = all datasets.",
    )
    dataset: str | None = Field(
        default=None,
        description="Deprecated: single dataset; prefer `datasets`.",
    )
    min_match_ratio: float | None = Field(
        default=None,
        description="With taxonomy filters: require matched_tasks/task_count >= this (0-1).",
    )


class ExportBody(BaseModel):
    repo_ids: list[str]
    format: Literal["json", "python_list"] = "json"


def create_app(
    *,
    db_path: Path | None = None,
    taxonomy_path: Path | None = None,
    static_dir: Path | None = None,
) -> Any:
    db_path = db_path or Path(os.environ.get("CPT_INDEX_DB", "tools/cpt_dataset_selector/data/index.sqlite3"))
    taxonomy_path = taxonomy_path or Path(os.environ.get("CPT_TAXONOMY", "")) or None
    static_dir = static_dir or Path(__file__).resolve().parent / "static"

    tax = load_taxonomy(taxonomy_path if taxonomy_path and taxonomy_path.is_file() else None)

    app = FastAPI(title="CPT Dataset Selector", version="0.1.0")

    # Return str — never annotate route params as pathlib.Path: FastAPI treats Path as a
    # query/path parameter source and ignores Depends, causing "Field required" for query.index_db.
    def require_db() -> str:
        if not db_path.is_file():
            raise HTTPException(status_code=503, detail=f"Index database not found: {db_path}")
        return str(db_path.resolve())

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/datasets")
    def list_datasets(index_db: str = Depends(require_db)) -> dict[str, Any]:
        conn = sqlite3.connect(index_db)
        ensure_repos_robot_type_column(conn)
        conn.commit()
        cur = conn.execute(
            """
            SELECT DISTINCT family FROM repos
            WHERE family IS NOT NULL AND TRIM(family) != ''
            ORDER BY family
            """
        )
        names = [r[0] for r in cur.fetchall()]
        conn.close()
        return {"datasets": names}

    @app.get("/api/taxonomy")
    def get_taxonomy() -> dict[str, Any]:
        return {
            "schema_version": tax.schema_version,
            "reference": tax.robocoin_reference,
            "structured_field_aliases": {k: list(v) for k, v in tax.structured_field_aliases.items()},
            "atomic_actions": [{"id": o.id, "label": o.label} for o in tax.atomic_actions],
            "object_categories": [{"id": o.id, "label": o.label} for o in tax.object_categories],
            "scenes": [{"id": o.id, "label": o.label} for o in tax.scenes],
        }

    @app.post("/api/query")
    def post_query(
        body: Annotated[QueryBody, Body()],
        index_db: str = Depends(require_db),
    ) -> dict[str, Any]:
        if body.min_match_ratio is not None and not (0.0 <= body.min_match_ratio <= 1.0):
            raise HTTPException(status_code=422, detail="min_match_ratio must be between 0 and 1")
        ds_names: list[str] = []
        for s in body.datasets:
            t = (s or "").strip()
            if t:
                ds_names.append(t)
        if not ds_names and body.dataset:
            d = (body.dataset or "").strip()
            if d:
                ds_names = [d]
        q = QueryFilter(
            atomic_actions=tuple(body.atomic_actions),
            object_categories=tuple(body.object_categories),
            scenes=tuple(body.scenes),
            include_incomplete_meta=body.include_incomplete_meta,
            dataset_filters=tuple(ds_names),
            min_match_ratio=body.min_match_ratio,
        )
        rows = query_repos(index_db, q, taxonomy=tax)
        total_episodes = 0
        total_duration_hours = 0.0
        for r in rows:
            te = r.get("total_episodes")
            if te is not None:
                with contextlib.suppress(TypeError, ValueError):
                    total_episodes += int(te)
            dh = r.get("duration_hours")
            if dh is not None:
                with contextlib.suppress(TypeError, ValueError):
                    total_duration_hours += float(dh)
        return {
            "count": len(rows),
            "repos": rows,
            "total_episodes": total_episodes,
            "total_duration_hours": round(total_duration_hours, 4),
        }

    @app.get("/api/repo/{repo_id:path}")
    def get_repo(repo_id: str, index_db: str = Depends(require_db)) -> dict[str, Any]:
        conn = sqlite3.connect(index_db)
        ensure_repos_robot_type_column(conn)
        conn.commit()
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM repos WHERE repo_id = ?", (repo_id,))
        r = cur.fetchone()
        if not r:
            conn.close()
            raise HTTPException(status_code=404, detail="repo not found")
        cur = conn.execute(
            "SELECT line_no, task_index, raw_json, normalized_text FROM tasks WHERE repo_id = ?",
            (repo_id,),
        )
        tasks = [dict(x) for x in cur.fetchall()]
        conn.close()
        repo = dict(r)
        if "family" in repo:
            repo["dataset"] = repo.pop("family")
        return {"repo": repo, "tasks": tasks}

    @app.get("/api/search")
    def search(
        index_db: str = Depends(require_db),
        q: str = "",
        limit: int = 200,
    ) -> dict[str, Any]:
        rows = fts_search(index_db, q, limit=limit)
        return {"count": len(rows), "hits": rows}

    @app.post("/api/export")
    def export_selection(body: ExportBody) -> dict[str, str]:
        if body.format == "json":
            return {"content": json.dumps(body.repo_ids, ensure_ascii=False, indent=2)}
        lines = [f'    "{r}",' for r in body.repo_ids]
        inner = "\n".join(lines)
        content = f"REPO_ID = [\n{inner}\n]\n"
        return {"content": content}

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def index() -> Any:
            return FileResponse(
                str(static_dir / "index.html"),
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate",
                    "Pragma": "no-cache",
                },
            )

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("CPT_HOST", "0.0.0.0")
    port = int(os.environ.get("CPT_PORT", "9897"))
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
