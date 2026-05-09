"""Local graph/memory backend used when Zep Cloud is disabled.

This is a lightweight filesystem-backed adapter. It keeps MiroFish's graph
contract available for local PoC runs without sending source material to Zep.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


class LocalQMDGraphStore:
    def __init__(self, root_dir: Optional[str] = None):
        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.root_dir = root_dir or os.path.join(app_root, "uploads", "local_graphs")
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, graph_id: str) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", graph_id)
        return os.path.join(self.root_dir, f"{safe_id}.json")

    def _load(self, graph_id: str) -> Dict[str, Any]:
        path = self._path(graph_id)
        if not os.path.exists(path):
            raise ValueError(f"Local graph not found: {graph_id}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, graph: Dict[str, Any]) -> None:
        with open(self._path(graph["graph_id"]), "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

    def create_graph(self, name: str) -> str:
        graph_id = f"local_qmd_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        self._save({
            "graph_id": graph_id,
            "name": name,
            "description": "MiroFish local_qmd graph",
            "created_at": now,
            "ontology": {},
            "episodes": [],
            "nodes": [],
            "edges": [],
        })
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        path = self._path(graph_id)
        if os.path.exists(path):
            os.remove(path)

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        graph = self._load(graph_id)
        graph["ontology"] = ontology or {}
        self._save(graph)

    def add_text_batches(self, graph_id: str, chunks: List[str]) -> List[str]:
        graph = self._load(graph_id)
        episode_ids = []
        for chunk in chunks:
            episode_id = f"episode_{uuid.uuid4().hex[:16]}"
            episode_ids.append(episode_id)
            graph["episodes"].append({
                "uuid": episode_id,
                "type": "text",
                "data": chunk,
                "processed": True,
            })
            self._merge_extracted_graph(graph, chunk, episode_id)
        self._save(graph)
        return episode_ids

    def get_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        return self._load(graph_id).get("nodes", [])

    def get_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        return self._load(graph_id).get("edges", [])

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        graph = self._load(graph_id)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> Dict[str, List[Dict[str, Any]]]:
        query_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 1]

        def score(text: str) -> int:
            lowered = (text or "").lower()
            if not lowered:
                return 0
            value = 100 if query.lower() in lowered else 0
            value += sum(10 for term in query_terms if term in lowered)
            return value

        nodes = []
        edges = []
        if scope in {"nodes", "both"}:
            nodes = sorted(
                self.get_nodes(graph_id),
                key=lambda n: score(f"{n.get('name', '')} {n.get('summary', '')}"),
                reverse=True,
            )
            nodes = [n for n in nodes if score(f"{n.get('name', '')} {n.get('summary', '')}") > 0][:limit]
        if scope in {"edges", "both"}:
            edges = sorted(
                self.get_edges(graph_id),
                key=lambda e: score(f"{e.get('name', '')} {e.get('fact', '')}"),
                reverse=True,
            )
            edges = [e for e in edges if score(f"{e.get('name', '')} {e.get('fact', '')}") > 0][:limit]
        return {"nodes": nodes, "edges": edges}

    def _entity_types(self, graph: Dict[str, Any]) -> List[str]:
        names = [
            item.get("name")
            for item in graph.get("ontology", {}).get("entity_types", [])
            if item.get("name")
        ]
        return names or ["Company", "Catalyst", "Risk", "InvestorPersona"]

    def _merge_extracted_graph(self, graph: Dict[str, Any], text: str, episode_id: str) -> None:
        existing_names = {node["name"].lower(): node for node in graph["nodes"]}
        entity_types = self._entity_types(graph)
        candidates = list(self._candidate_entities(text))
        added_nodes = []

        for index, candidate in enumerate(candidates[:40]):
            key = candidate["name"].lower()
            if key in existing_names:
                continue
            entity_type = entity_types[index % len(entity_types)]
            node = {
                "uuid": f"node_{uuid.uuid4().hex[:16]}",
                "name": candidate["name"],
                "labels": ["Entity", entity_type],
                "summary": candidate["summary"],
                "attributes": {
                    "source": "local_qmd",
                    "episode_id": episode_id,
                },
            }
            graph["nodes"].append(node)
            existing_names[key] = node
            added_nodes.append(node)

        if len(added_nodes) < 2:
            return

        for left, right in zip(added_nodes, added_nodes[1:]):
            graph["edges"].append({
                "uuid": f"edge_{uuid.uuid4().hex[:16]}",
                "name": "RELATED_TO",
                "fact": f"{left['name']} and {right['name']} were co-mentioned in local source context.",
                "source_node_uuid": left["uuid"],
                "target_node_uuid": right["uuid"],
                "attributes": {"source": "local_qmd", "episode_id": episode_id},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "valid_at": None,
                "invalid_at": None,
                "expired_at": None,
                "episodes": [episode_id],
            })

    def _candidate_entities(self, text: str) -> Iterable[Dict[str, str]]:
        seen = set()
        stopwords = {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "LLM", "API"}

        for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,5}\b", text):
            name = match.group(0)
            if name in stopwords or name in seen:
                continue
            seen.add(name)
            yield {"name": name, "summary": self._line_for(text, match.start())}

        for line in text.splitlines():
            cleaned = line.strip(" \t#-*0123456789.").strip()
            if not cleaned or len(cleaned) < 4:
                continue
            name = re.split(r"[:|,-]", cleaned, maxsplit=1)[0].strip()
            if len(name) > 80 or len(name.split()) > 8:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            yield {"name": name, "summary": cleaned[:500]}

    def _line_for(self, text: str, position: int) -> str:
        start = text.rfind("\n", 0, position) + 1
        end = text.find("\n", position)
        if end == -1:
            end = len(text)
        return text[start:end].strip()[:500]
