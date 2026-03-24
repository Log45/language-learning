If you already have JLPT-level vocab JSON, putting “just the kanji” into a vector DB is usually the least effective way to enforce level.

Why embeddings + “just kanji” is a bad fit

JLPT control is a constraint problem, not a “semantic similarity search” problem.

“Kanji-only” loses the things you’ll need to enforce/rewrite correctly:

readings (かな), lemma, POS, meaning, conjugation behavior

Vector search will happily return nearby items (semantically or orthographically), not necessarily the exact words you want, and it won’t reliably prevent out-of-level words.

Better setup: treat JLPT vocab as a structured whitelist

Keep your vocab in a fast structured store (dict/SQLite) keyed by lemma, with fields:
{lemma, reading, level, pos, meanings, example(s), alt_forms}

Then use one of these two patterns (often both):

Option A: “Whitelist + validator + rewrite” (most reliable)

Model generates text.

You run a Japanese tokenizer (SudachiPy or MeCab) to get lemmas.

If any lemma is not in the allowed set for the target JLPT level → ask the model to rewrite, providing a list of violating lemmas + optional substitutions.

This guarantees level control regardless of what the model tries to do.

Why this is the one I’d pick first: it’s deterministic and doesn’t depend on retrieval quality.

Option B: “Vocab-picker tool” (strong + simple in LangChain)

Instead of embedding all vocab, create a tool like:

get_allowed_vocab(level, topic, n) → returns a curated list of allowed lemmas + readings + gloss

suggest_substitutes(level, forbidden_lemmas) → returns easier replacements

Then your agent does:

pick vocab (tool)

generate using that vocab

optional validator pass

This is “RAG-like” in spirit (model consults external knowledge) but it’s structured retrieval, not vector search.

If you still want a retriever: use metadata filters and richer docs

If you insist on LangChain retrievers, do not store “just kanji”.

Store each entry as a Document like:

page_content: "食べる (たべる) — to eat"

metadata: {"level": "N5", "pos": "verb", "lemma": "食べる", "reading": "たべる" ...}

Then retrieve with:

metadata filter: level <= target_level (or exact level set)

and search by topic (English prompt / Japanese topic words)

But even then, this is better for suggesting level-appropriate vocab, not enforcing.

Recommended architecture for your exact case (JLPT + JSON)
1) Keep JSON as the source of truth (not the vector DB)

Load into an in-memory set/map like:

allowed_lemmas_by_level["N5"] = {...}

allowed_lemmas_by_level["N4"] = allowed["N5"] ∪ N4 (if you want cumulative)

2) Use your vector DB for “content”, not “the whitelist”

Use embeddings for things like:

example sentences

short graded passages

explanations/notes you wrote

minimal pairs / common mistakes

Those benefit from semantic retrieval.

3) Enforce output with a validator loop

detect out-of-level lemmas

rewrite with constraints

This gets you the control you want with far less pain than trying to make the vector DB behave like a whitelist.

Concrete suggestions to improve what you have right now

If you keep the vector DB approach temporarily:

Include reading + meaning + lemma in the page_content

Put {"jlpt": "N5"} in metadata

When you “retrieve for level”, don’t similarity-search the entire vocab DB.

Instead: retrieve only from that JLPT level (metadata filter), and treat retrieval as a “suggested vocab list”.

But I’d still add the validator. That’s the part that makes it robust.

If you want, paste one sample JSON entry (or your schema) and tell me whether you want the allowed vocab to be cumulative (N5 words allowed in N4 outputs). I’ll sketch a clean data structure + a validator/rewrite loop that plugs into your current LangChain agent setup.