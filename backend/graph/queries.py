CREATE_CONSTRAINTS = [
    "CREATE CONSTRAINT word_id_unique IF NOT EXISTS FOR (w:Word) REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT kanji_character_unique IF NOT EXISTS FOR (k:Kanji) REQUIRE k.character IS UNIQUE",
    "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT meaning_text_unique IF NOT EXISTS FOR (m:Meaning) REQUIRE m.text IS UNIQUE",
    "CREATE CONSTRAINT reading_unique IF NOT EXISTS FOR (r:Reading) REQUIRE (r.value, r.type) IS UNIQUE",
]

UPSERT_WORD = """
MERGE (w:Word {id: $id})
SET w.kanji = $kanji,
    w.kana = $kana,
    w.romaji = $romaji,
    w.jlptLevel = $jlpt_level,
    w.isCommon = $is_common
"""

UPSERT_WORD_MEANING = """
MERGE (w:Word {id: $word_id})
MERGE (m:Meaning {text: $meaning})
MERGE (w)-[:HAS_MEANING]->(m)
"""

UPSERT_WORD_TAG = """
MERGE (w:Word {id: $word_id})
MERGE (t:Tag {name: $tag_name})
MERGE (w)-[:HAS_TAG]->(t)
"""

UPSERT_WORD_READING = """
MERGE (w:Word {id: $word_id})
MERGE (r:Reading {value: $reading, type: 'word'})
MERGE (w)-[:HAS_READING]->(r)
"""

UPSERT_KANJI = """
MERGE (k:Kanji {character: $character})
SET k.meaning = $meaning,
    k.onyomi = $onyomi,
    k.kunyomi = $kunyomi
"""

UPSERT_WORD_CONTAINS_KANJI = """
MERGE (w:Word {id: $word_id})
MERGE (k:Kanji {character: $character})
MERGE (w)-[r:CONTAINS]->(k)
SET r.position = $position
"""

UPSERT_KANJI_READING = """
MERGE (k:Kanji {character: $character})
MERGE (r:Reading {value: $reading, type: $reading_type})
MERGE (k)-[hr:HAS_READING]->(r)
SET hr.source = $source
"""

UPSERT_KANJI_COOCCURS = """
MATCH (w:Word {id: $word_id})-[:CONTAINS]->(k1:Kanji)
MATCH (w)-[:CONTAINS]->(k2:Kanji)
WHERE k1.character <> k2.character
MERGE (k1)-[r:CO_OCCURS_WITH]->(k2)
ON CREATE SET r.count = 1
ON MATCH SET r.count = r.count + 1
"""

RELATED_WORDS_BY_KANJI = """
MATCH (w:Word {id: $word_id})-[:CONTAINS]->(k:Kanji)<-[:CONTAINS]-(related:Word)
WHERE related.id <> $word_id
RETURN related.id AS id, related.kanji AS kanji, related.kana AS kana, count(k) AS overlap
ORDER BY overlap DESC
LIMIT $limit
"""

READINGS_FOR_WORD = """
MATCH (w:Word {id: $word_id})-[:HAS_READING]->(r:Reading)
RETURN r.value AS reading
ORDER BY r.value
"""

KANJI_FOR_WORD = """
MATCH (w:Word {id: $word_id})-[c:CONTAINS]->(k:Kanji)
RETURN k.character AS character, c.position AS position
ORDER BY c.position
"""

# Bounded neighborhood for visualization (no APOC). Caller supplies allowed rel types + edge cap.
SUBGRAPH_FROM_WORD = """
MATCH (center:Word {id: $word_id})
CALL (center) {
  MATCH (center)-[r]-(n)
  WHERE type(r) IN $allowed_types
  WITH r, n
  LIMIT $edge_limit
  RETURN collect({r: r, n: n}) AS pairs
}
WITH center, pairs
WITH center, CASE WHEN size(pairs) = 0 THEN [null] ELSE pairs END AS rows
UNWIND rows AS pair
RETURN center,
       CASE WHEN pair IS NULL THEN null ELSE pair['r'] END AS r,
       CASE WHEN pair IS NULL THEN null ELSE pair['n'] END AS n
"""

SUBGRAPH_FROM_KANJI = """
MATCH (center:Kanji {character: $character})
CALL (center) {
  MATCH (center)-[r]-(n)
  WHERE type(r) IN $allowed_types
  WITH r, n
  LIMIT $edge_limit
  RETURN collect({r: r, n: n}) AS pairs
}
WITH center, pairs
WITH center, CASE WHEN size(pairs) = 0 THEN [null] ELSE pairs END AS rows
UNWIND rows AS pair
RETURN center,
       CASE WHEN pair IS NULL THEN null ELSE pair['r'] END AS r,
       CASE WHEN pair IS NULL THEN null ELSE pair['n'] END AS n
"""

