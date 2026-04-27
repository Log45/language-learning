# Neo4j Browser: exploring the Japanese knowledge graph

## Connect

1. Start services: `docker compose up -d` (from repo root).
2. Open **Neo4j Browser**: [http://localhost:7474](http://localhost:7474).
3. Use the same credentials as [docker-compose.yml](../../docker-compose.yml) / [backend/db/neo4j_session.py](../db/neo4j_session.py):
   - **URI**: `bolt://localhost:7687` (Browser may default to `neo4j://localhost:7687`; either works for local).
   - **Username**: `neo4j`
   - **Password**: `password` (unless you overrode `NEO4J_AUTH`).

Override in app code with env: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

## Bounded queries (keep `LIMIT` small)

Large `MATCH` patterns can freeze the tab. Prefer **word id** or **single kanji** anchors and tight limits.

### Word neighborhood (1 hop)

Replace `123` with a real `Word.id` from your DB:

```cypher
MATCH (w:Word {id: 123})-[r]-(n)
RETURN w, r, n
LIMIT 120;
```

### Kanji and words that contain it

```cypher
MATCH (k:Kanji {character: '生'})<-[:CONTAINS]-(w:Word)
RETURN k, w
LIMIT 50;
```

### Kanji readings (via `HAS_READING`)

```cypher
MATCH (k:Kanji {character: '生'})-[hr:HAS_READING]->(r:Reading)
RETURN k, hr, r
LIMIT 40;
```

### Co-occurrence slice (dense; keep limit low)

```cypher
MATCH (k:Kanji {character: '生'})-[c:CO_OCCURS_WITH]-(k2:Kanji)
RETURN k, c, k2
ORDER BY c.count DESC
LIMIT 30;
```

### Paths between two kanji (optional; expensive if unrestricted)

```cypher
MATCH (a:Kanji {character: '学'}), (b:Kanji {character: '生'})
MATCH p = shortestPath((a)-[:CO_OCCURS_WITH*..6]-(b))
RETURN p
LIMIT 5;
```

## Programmatic export

- **PyVis HTML**: [../scripts/export_graph_html.py](../scripts/export_graph_html.py) — writes a standalone interactive file. From repo root with `PYTHONPATH` set to the project root (parent of `backend`). Use **hyphenated** flags (`--word-id`, not `--word_id`):

  `PYTHONPATH=. python backend/scripts/export_graph_html.py --word-id 1 --out /tmp/word_1.html`

  Or as a module:

  `PYTHONPATH=. python -m backend.scripts.export_graph_html --word-id 1 --out /tmp/word_1.html`

  Kanji-centered:

  `PYTHONPATH=. python backend/scripts/export_graph_html.py --kanji 生 --out /tmp/kanji.html`

  Optional: hide dense edges, e.g. `--exclude-rel HAS_TAG,CO_OCCURS_WITH`

- **JSON API**: `GET /graph/subgraph` on the FastAPI app — same subgraph shape for custom UIs (e.g. `?word_id=1&edge_limit=80`).
