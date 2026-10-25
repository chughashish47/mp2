# MP2 Reflection

## What worked

Paragraph-aware chunking worked well. Splitting on blank lines and detecting short
unpunctuated lines as section headers kept each chunk semantically coherent — the
retrieved sections consistently matched the topic of the question. All 5 questions
passed the source-match check on the first attempt, and the citation headers
(`[Source: title — section]`) in the context made the LLM reliably attribute its
answers to the right story.

## What didn't work

The facts-matched count was low on questions q4 and q5 even though the answers were
correct. The validator does a substring match, so "stamping fake coins" doesn't match
"counterfeit" and "fled by the five-fifteen train" doesn't match "five-fifteen train".
The answers weren't wrong — they were just phrased differently from the expected facts
strings. More careful phrasing in the expected_facts list would have made this a more
useful diagnostic.

## What I'd change

I'd try smaller chunks (around 300 characters) for multi-hop questions like q5. The
Irene Adler escape spans several scenes — her recognising Holmes in two disguises, then
following him back to Baker Street, then fleeing from Charing Cross — and all of that
is spread across different paragraphs. At 500 characters, some of those details end up
in different chunks and k=3 doesn't always pull all of them. Either smaller chunks or
k=5 would give the LLM more complete context for that kind of question.

## One surprise

I didn't expect dependency version drift to be the main source of friction. The starter
template used `qdrant.search()` and `recreate_collection()`, both of which were removed
or deprecated in `qdrant-client 1.20.0`. Updating to `query_points()` and
`collection_exists()` + `create_collection()` was a two-line fix each, but it was a
good reminder that pinning versions in `requirements.txt` matters — especially when
your project might be run months after it was written.
