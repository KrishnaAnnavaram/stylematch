# The writing standard: ASD-STE100 Simplified Technical English for stylematch

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **catalog** | The table of products that stylematch indexes. | dataset (alone), inventory, store |
| **product** | One row of the catalog, with one product ID. | item (in prose), article, SKU |
| **product ID** | The unique key of a product (`product_id`). | SKU, code |
| **query** | The free text that a shopper types. | prompt, search string, question |
| **intent** | The parsed form of a query: gender, price range, categories, colours, occasions, expansion terms. | slots, entities |
| **hard filter** | A filter that removes products from the pool: gender and price. | constraint, rule |
| **occasion** | An event in a query, for example `party` or `wedding`. | use case, event type |
| **expansion terms** | The product words that the occasion map adds to the search text. | synonyms, related words |
| **pool** | The products that pass the hard filters for one query. | candidate set (for the filtered catalog), subset |
| **index** | The saved TF-IDF vectors, BM25 counts and dense vectors in the index folder. | vector store, database, cache |
| **manifest** | The file `manifest.json` in the index, with the fingerprint. | metadata file |
| **fingerprint** | A SHA-256 value of the normalized catalog, the embedder and the index version. | hash (alone), checksum |
| **embedder** | The component that changes text into a vector: `lsa`, `openai` or `sentence-transformers`. | encoder, embedding model (in prose) |
| **method** | A retrieval method: `tfidf`, `bm25`, `dense` or `hybrid`. | algorithm, engine |
| **candidate** | A product that a method returns from the pool, with a retrieval score. | result, match, hit |
| **retrieval score** | The score that the method gives a candidate. | rating, relevance (for a score) |
| **fusion** | Reciprocal-rank fusion (RRF) of the BM25 list and the dense list. | merge, blend |
| **re-ranker** | The component that orders the candidates again: `none`, `attribute` or `llm`. | ranker, sorter |
| **chat model** | The optional LLM behind the `llm` re-ranker. | GPT (in prose), AI, bot |
| **reason** | The short text that tells why a product fits the query. | explanation (in tables only), justification |
| **system** | One method plus one re-ranker, as the evaluation names it, for example `hybrid_attribute`. | model (for a pipeline), variant |
| **baseline** | The `tfidf` system. All comparisons use it. | reference model |
| **gold set** | The labelled queries, with a grade for each relevant product. | test set, ground truth (in prose) |
| **grade** | The relevance label of a product for a query: 0, 1 or 2. | score, rating |
| **labelling sheet** | The blind CSV file from `stylematch pool` that annotators fill. | annotation file |
| **study** | The blinded, randomized user study. | survey, A/B test (in prose) |
| **sheet** | The blind study file that participants fill. | form, questionnaire |
| **key** | The hidden study file that maps each list to its system. | answer file, mapping |
| **participant** | A person who rates lists in the study. | user (in the study), rater |
| **rating** | A score from 1 to 10 that a participant gives a list in the study. | grade, score (for a study value) |
| **verdict** | The study result text that the analysis calculates from the tests. | conclusion, finding |
| **synthetic data** | The generated catalog and gold set of `stylematch synth`. They are not real data. | fake data, mock data, sample data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read a file and check it against its column contract. |
| **normalize** | Change the catalog columns to their standard names, types and values. |
| **build** | Make a new index from the catalog and save it. |
| **reuse** | Load the saved index because the fingerprint did not change. |
| **parse** | Change a query into an intent. |
| **filter** | Remove the products that do not pass a hard filter. |
| **retrieve** | Get the candidates for a query with one method. |
| **fuse** | Make one ranked list from the BM25 list and the dense list. |
| **re-rank** | Order the candidates again. |
| **validate** | Remove each product ID that is not a candidate, and remove repeats. |
| **evaluate** | Measure a system against the gold set. |
| **unblind** | Join the study ratings with the key. |
