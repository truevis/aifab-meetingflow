# Live flowchart model test and output recommendations

Reviewed September 20, 2026, using **`vids/sample transcript.txt`** as the input for every live test. Scope is what the four flowchart engines return and how that result helps a reader of the app. This review does not assess media transcription.

## Tests run

`python -m unittest discover -s tests -p 'test_*.py' -v` passed **5/5** validator tests. The four live scripts, `python tests/test_{jev,mercury,glm,gemini}_sample.py`, each completed successfully and wrote its JSON report in `tests/`. “Successful” here means a response parsed and passed structural validation; it does **not** mean its process claims were accurate.

| App engine | Time | Nodes / edges / lanes | Script expectation misses | Main observation |
| --- | ---: | ---: | ---: | --- |
| Jev hybrid | 30.3 s | 5 / 4 / 2 | 1 | Clearer than the earlier line-by-line Jev sample, but petition is missing and timing is overstated. |
| Mercury | 21.8 s | 5 / 4 / 2 | 2 | Shortest run, but it misses the petition and treats conflicting timing as a settled edge. |
| GLM | 74.3 s | 6 / 5 / 2 | 0 | Captures the petition as an alternative and raises useful questions, yet only returns 21 utterances from 98 input segments. |
| Gemini 3.8 Flash | 39.6 s | 6 / 5 / 3 | 0 | Best detail on the main sequence and roles in this one run, yet still presents disputed deadlines as rules and invents a completed endpoint. |

The reports are [`test_jev_sample.json`](../tests/test_jev_sample.json), [`test_mercury_sample.json`](../tests/test_mercury_sample.json), [`test_glm_sample.json`](../tests/test_glm_sample.json), and [`test_gemini_sample.json`](../tests/test_gemini_sample.json). This is one transcript and one run per engine, so the times and quality ranking are observations, not a general benchmark. Jev is explicitly a **Mercury synthesis plus Jev graph check** in `analyze_transcript_with_jev`; its output should not be read as a Jev-only generation result.

## What each engine should return differently

### Jev hybrid

- **Keep:** Five readable nodes, an appointment alternative, and a question about notice timing. The Jev route now checks candidate relationships instead of building a graph from caption order.
- **Improve:** The result says `Choose Method (Appointment vs. Special Election)` on the main path even though the speakers favor the election and reaffirm that at the end. `Warn Meeting (30-day rule)` compresses several distinct, disputed timing claims into one apparently settled rule. `Hold Special Election` has type `end` although the meeting has not happened.
- **Most useful return:** A board plan with `Set date` and `Issue notice` as planned actions; appointment as an alternative; the petition as a separate, unchosen option; timing questions that cite the competing source segments. Jev's edge checks should surface rejected relationships and the source excerpt that led to rejection, without turning low-confidence checks into unexplained disconnected nodes.
- **Specific evidence gap:** No petition appears in either `alternatives` or `confirmations`, despite source segments **U56** and **U67**. The two-source model pipeline should pass these salient alternatives through the Jev check unchanged unless Jev is explicitly asked to verify them.

### Mercury

- **Keep:** A compact graph, a separate appointment alternative, and useful board/voter lanes.
- **Improve:** The `N4 → N5` edge is labeled `30-45 day wait` and cites **U73**, which says **30–40**; **U75** contains the later “30 and 45” phrase. The output has a generic date question, but no question that acknowledges this conflict. `Identify Vacancy and Appointees` and `Decide Filling Method` consume diagram space without making the chosen work clearer. One confirmation asks whether the election will consider “diversity in the results,” which is not a concrete process ambiguity to resolve.
- **Most useful return:** A compact 4–6 step *planned* election path, with `Set meeting date`, `Issue official notice`, and `Hold election` as distinct steps. Return both timing claims as sourced facts with an unresolved confirmation, and include the petition as an alternative or unresolved contingency.
- **Specific generation instruction:** Never put a contested number into a node or edge label as if verified. When citations conflict, place the number in a confirmation with the competing source IDs.

### GLM Flash latest

- **Keep:** Both appointment and petition in `alternatives`, plus a question asking to verify the claimed statutory timeline. The graph also includes the two-seat scope.
- **Improve:** It returns only **21** `utterances` for **98** source segments. `app.py` stores any nonempty returned utterance list, so choosing GLM can replace the fuller transcript feed with a condensed one. Its `Decide filling method: appoint vs. special election` node still depicts a choice that speakers have largely resolved. The notice node asserts a **60-day** and **30–40-day** rule while its own confirmation asks for verification. `New member(s) seated` is a future assumption, not an established step. The petition alternative cites **U67** correctly, but also cites unrelated **U13** and **U14**, which concern Pam's seat.
- **Most useful return:** Preserve the same utterance count and IDs as the input, or return only annotations keyed by source ID while the app retains the original feed. Separate the claimed timing from the diagram label. Require accurate evidence IDs for alternatives and omit the unsupported seating endpoint.
- **Specific model contract:** “Return exactly one annotation for every provided source segment, in order; do not summarize or merge entries in the utterance log. Synthesize the graph separately.” Validate this at the caller if the model continues to produce the log.

### Gemini 3.8 Flash

- **Keep:** It captures the board's chosen direction, calendar coordination, notice, two-seat scope, voter role, and both alternatives in a readable sequence. It correctly cites **U56** and **U67** for the petition alternative. This is the strongest starting graph of these four individual runs.
- **Improve:** The notice node says `within 60 days of vacancy`, while the transcript speaker says **60 days from resignation to warn**; those reference events must remain distinct. The meeting node asserts **30–40 days** while the output also asks whether **U75**'s “45” conflicts. `Candidates submit candidacy for the two available seats` infers a formal process from statements that two people expressed interest and others could step forward. The `Special election concluded` endpoint implies completion of a future event. The `Town Voters` lane owns a `Convene special town meeting` action, although convening and voting are different responsibilities.
- **Most useful return:** Keep its calendar → notice → election skeleton and two-seat context, but make candidate participation background unless a submission procedure is stated. Split actor ownership: board or an explicitly identified official issues notice; voters vote. Keep the final event planned and stop the graph at that event. Put disputed notice and petition timing in confirmations, with clear reference events.

## Shared improvements in the current code

1. **Require agreement between fields.** In `utils/meeting_analyzer.py`, the schema allows `status` on nodes and edges, but the validator only checks that its value belongs to an enum. It does not catch “planned” nodes that describe completed results, an `adopted` edge asserting a legal requirement, or a disputed deadline in a label beside an unresolved confirmation. Add a review step that compares assertions in labels, facts, edge labels, and confirmations before caching.
2. **Keep the complete transcript feed.** The model's `utterances` array currently replaces the app's cached feed when nonempty (`app.py`, `_store_engine_analysis`). The graph model can return a shortened log, as GLM did. Preserve the parsed source segments locally and merge model annotations by `source_id` (or reject missing/duplicate IDs). Keep the existing feed behavior for every engine.
3. **Score citation relevance, not just existence.** `validate_workflow_graph` verifies that a `source_id` exists, but GLM's petition citation shows that an existing ID can support a different topic. For this fixture, add a few human-reviewed source anchors and report claims whose cited snippets do not address the label. Avoid claiming that the structural validator establishes factual support.
4. **Make `alternatives`, `facts`, and status visible in a compact way.** The app shows alternative labels, but stores neither returned `facts` nor evidence links, and its step table shows neither status nor citations. The result would be more informative if a reader could see *planned*, *adopted*, or *unresolved* beside each step and inspect its source timestamps. This is most valuable for the disputed timing and two-seat claims. Keep the primary diagram concise.
5. **Use confirmations for actionable uncertainty.** The existing confirmation cards present `confidence` as a percent. All four outputs include model-generated values, but these are not calibrated probabilities; a 90% number beside an unresolved deadline may mislead. Prefer a question plus the two short competing claims and source timestamps. Rank questions by whether the answer changes the workflow.
6. **Evaluate process truth beyond keyword checks.** `tests/town_board_eval.py` reports zero misses for GLM and Gemini while their outputs still contain unsupported endpoints and contradictory timing claims. Its current checks inspect label keywords, rough node count, and whether a petition appears anywhere. Add criteria for citation accuracy, preserved transcript count, planned-versus-completed wording, owner/action agreement, and whether disputed numbers are asserted as rules. Use human review for the semantic parts.
7. **Trim response work where safe.** The strict schema currently asks every flowchart model for a full `translated_transcript`, a detailed utterance log, facts, alternatives, and a graph. For an English input, the full transcript is already local. Consider a small graph-focused response plus keyed annotations, then retain the existing transcript data in the app. Measure quality and cost before making this split; do not lose the working translation behavior for non-English inputs.

## Concrete Python suggestions

The first check below makes a real failure in this run visible without requiring a new model call. Add it near `run_engine_sample` in `tests/town_board_eval.py` and append the returned messages to `expectation_misses`. It checks preservation of the feed, not diagram quality:

```python
def score_utterance_coverage(payload: dict[str, Any], source_count: int) -> list[str]:
    """Flag a model log that would replace the app feed with fewer entries."""
    utterances = payload.get("utterances")
    if not isinstance(utterances, list):
        return ["Model did not return an utterance list."]
    if len(utterances) != source_count:
        return [
            f"Utterance coverage {len(utterances)}/{source_count}; "
            "the app may show a shortened transcript feed."
        ]
    return []
```

Call it with `len(_workflow_source_segments(transcript))` from the already loaded sample. The test helper would need to import `_workflow_source_segments` or expose a public equivalent. For this run the check would report GLM's **21/98** result.

The following proposed caller guard keeps the full feed while still displaying model annotations. It belongs in a function in `app.py` and should be used inside `_store_engine_analysis` when a source utterance list is available. Existing `text`, `speaker`, and `timestamp` stay authoritative; model fields annotate them:

```python
def merge_utterance_annotations(
    source_utterances: list[dict[str, Any]],
    model_utterances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep every source turn and add annotations from matching model IDs."""
    source_ids = [item.get("id") for item in source_utterances]
    model_ids = [
        item.get("id") if isinstance(item, dict) else None
        for item in model_utterances
    ]
    if (
        len(source_ids) != len(model_ids)
        or len(set(model_ids)) != len(model_ids)
        or set(source_ids) != set(model_ids)
    ):
        return [dict(item) for item in source_utterances]
    by_id = {
        item.get("id"): item
        for item in model_utterances
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    annotation_fields = (
        "is_business", "relevance_score", "action_type",
        "detected_step", "detected_role",
    )
    merged = []
    for source in source_utterances:
        item = dict(source)
        annotation = by_id.get(item.get("id"), {})
        if str(annotation.get("text") or "") != str(item.get("text") or ""):
            merged.append(item)
            continue
        for field in annotation_fields:
            if field in annotation:
                item[field] = annotation[field]
        merged.append(item)
    return merged
```

This requires the caller to preserve the parsed source utterances and their IDs through analysis. The current GLM output's 21 entries may summarize multiple segments, so **do not** map those summaries to source turns by list position or by a partial run of IDs. If the counts, IDs, or per-entry text do not match, ignore those annotations and retain the full source feed. For translated meetings, keep the existing translated text flow and decide how to align translated turns before merging; exact source-text equality is intended for the English test case.

For this sample, improve `score_town_board_expectations` with a small, explicit evidence check instead of another broad keyword search. A human-reviewed fixture can define acceptable source IDs for a claim:

```python
def score_petition_evidence(payload: dict[str, Any]) -> list[str]:
    """Check the fixture's known petition excerpts, not arbitrary IDs."""
    petition_sources = {"U56", "U67"}
    unrelated_sources = {"U13", "U14"}
    for item in payload.get("alternatives") or []:
        if not isinstance(item, dict):
            continue
        if "petition" not in str(item.get("label") or "").casefold():
            continue
        cited = set(item.get("source_ids") or [])
        misses = []
        if not cited & petition_sources:
            misses.append("Petition alternative cites no petition excerpt (U56 or U67).")
        if cited & unrelated_sources:
            misses.append("Petition alternative cites the unrelated seat discussion (U13/U14).")
        return misses
    return ["Petition alternative is missing."]
```

This fixture-specific check would expose GLM's extra, unrelated petition citations and the missing petition in Jev and Mercury. It is not a generic proof that the alternative's full wording is correct. Human review should still inspect claims about election rules and timing.

No application code was changed as part of this review. The live test scripts wrote their JSON reports under `tests/`; no browser test was run.
