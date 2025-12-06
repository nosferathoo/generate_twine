#!/usr/bin/env python3
import argparse
import sys
import json
import re
from collections import defaultdict, deque

import requests


# ============================================================
#  ULTRA-SKOMPRESOWANE PROMPTY
# ============================================================

def build_common_instructions(min_passages: int) -> str:
    """
    VERY highly compressed ruleset, with explicit location + state-token naming.
    """
    return f"""
You are an expert Twine/Twee interactive fiction author.

RULES (ESSENTIAL):
- Output ONLY valid Twee code (no comments, no markdown).
- Story language = description language.
- NO variables, NO macros, NO <<if>>, <<set>>, scripting.
- State is encoded ONLY in passage names + link graph.
- Passage names MUST NOT contain '_' anywhere.
- Must have ≥ {min_passages} passages.

PASSAGE NAME STRUCTURE (LOCATION + STATE TOKENS):
- A passage name has the following form:

    LocationName[-StateToken1-StateToken2-...]

- The FIRST part (before the first '-') is the LOCATION / SITUATION:
  - e.g. SkybreakerSublevel2, RelayStationCorridor, CityGateNight.
  - It MUST NOT contain '-' or '_'.
  - It should describe "where/when" the panel is.

- AFTER the location, you MAY append state tokens separated by '-':
  - Examples:
    - SkybreakerSublevel2-WithLyra-HasKeycard-JoinedFactionA
    - RelayStationCorridor-MetLyra-LostKeycard
    - CityGateNight-JoinedFactionB
  - State tokens are short flags encoding important facts:
    - MetX, WithX, LostX, LeftX         (characters)
    - HasItemY, LostItemY, LeftItemY    (items)
    - JoinedFactionA, BetrayedFactionB, NeutralFaction (factions)

- DO NOT use meaningless or neutral tokens such as:
  - Alone, EmptyInventory, Default, Normal, Generic, etc.
- ABSENCE of a positive token IS the default:
  - If there is no WithLyra / MetLyra token -> the player never met/is not with Lyra.
  - If there is no HasItemKeycard token -> the player does not carry the keycard.
- Use explicit negative tokens only when important and specific:
  - LostLyra, LeftKeycardBehind, BetrayedFactionA.

STATE CONSISTENCY & SYMMETRY:

- For any target passage X, several passages may link to X.
- Each passage name has:
  - a LOCATION part (first chunk, before the first '-'),
  - optional STATE TOKENS after the first '-'.

- When multiple passages link to the same target X:
  - Their STATE TOKENS must be consistent with the STATE TOKENS of X.
  - If two source passages share the same state (same STATE TOKENS overlap with X),
    they should behave symmetrically and lead to a compatible version of X.
  - If a source passage has a different state (different tokens overlap with X),
    you MUST NOT link it to the same X unless the text still perfectly matches
    that state.

- Asymmetry (INCONSISTENT links):
  - If many passages link to X but some of them have different or missing
    STATE TOKENS compared to X, treat those links as suspicious.
  - Fix asymmetry by either:
    - redirecting the suspicious sources to a more appropriate target passage
      whose name matches their STATE TOKENS, or
    - creating a separate variant of the target passage (e.g. a different
      LocationName-...-StateToken combination) and linking the appropriate
      sources there.
  - Do NOT leave sources that say "Alone" in text while linking to a passage
    whose name clearly encodes "With companions", or vice versa.

PASSAGE = PANEL:
- Each passage is ONE clear comic panel:
  - one location (LocationName),
  - one concrete moment,
  - one immediate situation.
- Time and travel happen BETWEEN passages, not as long summaries in one passage.
- Do NOT skip over major actions in a single passage: entering a new area,
  sneaking through danger, finding or stealing key items should take multiple passages.

KEY ITEMS / GOALS:
- Important items or objectives must NOT simply lie in the open waiting.
- Reaching a key item should require 2–3+ passages:
  - noticing hints or rumors,
  - approaching through danger or obstacles,
  - facing a guard, trap, lock, puzzle, or moral choice,
  - only then obtaining (or failing to obtain) the item.
- Never let the player "just pick up" a critical item in the same passage where
  it is first mentioned.

GRAPH RULES (CRITICAL):
- NO endless cycles, NO closed loops, NO ping-pong loops.
- Backtracking loops allowed only if ANY state can exit toward progress.
- EVERY passage must eventually reach an Ending.
- Endings MUST start with "Ending-" and have NO outgoing links.
- NO passage may become a dead-end unless it IS an Ending.

LINK RULES:
- Only format: [[Text->Passage]].
- NO spaces around ->.
- Each link must be on its own line (one line = one choice).
- All link targets MUST exist.

STORY:
- Must branch; ≥1 good ending, ≥3 bad endings.
- Choices must be in-world text (no meta "if you did X then click Y").
- No contradictions: cannot show characters/items never obtained on this route.

VERIFY BEFORE OUTPUT:
- All links valid and formatted correctly.
- No contradictions in text vs encoded state.
- No broken links.
- No non-ending dead-ends.
- No cycles without valid exits.
"""


def build_prompt(description: str, min_passages: int) -> str:
    return f"""
{build_common_instructions(min_passages)}

Create a full Twine/Twee story satisfying ALL rules above.

Description:
\"\"\"{description.strip()}\"\"\"
""".strip()


def build_fix_prompt(description: str, existing_twee: str, missing_passages, min_passages: int) -> str:
    missing_list = "\n".join(f"- {m}" for m in missing_passages)
    return f"""
{build_common_instructions(min_passages)}

Description:
\"\"\"{description.strip()}\"\"\"

Existing story:
\"\"\"{existing_twee}\"\"\"

Missing passages:
{missing_list}

Task:
- DO NOT modify or repeat existing passages.
- For EACH missing name output EXACTLY ONE Twee passage.
- NO new passage names beyond the missing list.
- Use valid Twee headers and links.
- Text must match world + state logic.

Output ONLY the missing passages.
""".strip()


def _patch_instructions(min_passages: int) -> str:
    """
    Fragment wspólny dla wszystkich promptów 'patchowych'.
    """
    common = build_common_instructions(min_passages)
    return f"""
{common}

IMPORTANT: YOU MUST RETURN ONLY A PATCH, NOT THE FULL STORY.

PATCH FORMAT:
- For a MODIFIED or NEW passage:
  - Output its FULL passage: header line and entire body, e.g.:
      :: PassageName
      (full body with text and links)

- For a DELETED passage:
  - Output ONLY its header line, e.g.:
      :: NameOfDeletedPassage
    with NO body (no text, no blank lines, nothing else).

- For a RENAMED passage:
  - Output one header line with the OLD name, with no body (delete):
      :: OldPassageName
  - AND one FULL passage with the NEW name:
      :: NewPassageName
      (full body)

RULES FOR PATCH:
- DO NOT output any passage that remains unchanged.
- DO NOT output comments, markdown, or explanations.
- Obey all naming, graph, and story rules from the common instructions.
"""


def build_cycles_fix_prompt(description: str,
                            existing_twee: str,
                            cycles,
                            min_passages: int) -> str:
    patch_common = _patch_instructions(min_passages)

    cycles_lines = []
    for i, cycle in enumerate(cycles, 1):
        arrow = " -> ".join(cycle + [cycle[0]]) if cycle else ""
        cycles_lines.append(f"- Cycle {i}: {arrow}")
    cycles_block = "\n".join(cycles_lines) if cycles_lines else "None"

    return f"""
{patch_common}

Description:
\"\"\"{description.strip()}\"\"\"

Current story:
\"\"\"{existing_twee}\"\"\"

STRUCTURAL PROBLEM (CYCLES ONLY):
- The following bad cycles were detected in the passage graph:
{cycles_block}

TASK:
- Fix ALL cycles listed above so that:
  - no player can get trapped in a closed loop with no exit,
  - every cycle has at least one way to exit toward progress and eventually reach an Ending,
    or is completely broken and replaced by acyclic structure.
- You MAY adjust links, rename passages, split passages, or delete them as needed.
- You SHOULD NOT significantly change the high-level narrative, only its structure.

OUTPUT:
- ONLY a patch (set of passage edits) in the PATCH FORMAT described above.
""".strip()


def build_dead_fix_prompt(description: str,
                          existing_twee: str,
                          dead_ends,
                          min_passages: int) -> str:
    patch_common = _patch_instructions(min_passages)

    dead_block = "\n".join(f"- {name}" for name in dead_ends) if dead_ends else "None"

    return f"""
{patch_common}

Description:
\"\"\"{description.strip()}\"\"\"

Current story:
\"\"\"{existing_twee}\"\"\"

STRUCTURAL PROBLEM (DEAD-ENDS ONLY):
- The following passages CANNOT reach any Ending- and are not themselves endings:
{dead_block}

TASK:
- For every passage above, modify the structure so that:
  - the player can eventually reach at least one Ending-,
  - non-ending passages never become terminal dead-ends.
- You MAY:
  - add new passages,
  - alter or redirect links,
  - rename or delete problematic passages,
  - as long as you respect the global rules and keep the narrative coherent.

OUTPUT:
- ONLY a patch (set of passage edits) in the PATCH FORMAT described above.
""".strip()


def build_asym_fix_prompt(description: str,
                          existing_twee: str,
                          asymmetries,
                          min_passages: int) -> str:
    patch_common = _patch_instructions(min_passages)

    if asymmetries:
        lines = []
        for target, probable, suspect in asymmetries:
            prob_s = ", ".join(probable) if probable else "none"
            susp_s = ", ".join(suspect) if suspect else "none"
            lines.append(
                f"- Target: {target}\n"
                f"    Probable correct sources: {prob_s}\n"
                f"    Suspect (inconsistent) sources: {susp_s}"
            )
        asym_block = "\n".join(lines)
    else:
        asym_block = "None"

    return f"""
{patch_common}

Description:
\"\"\"{description.strip()}\"\"\"

Current story:
\"\"\"{existing_twee}\"\"\"

STRUCTURAL PROBLEM (ASYMMETRIC CONVERGENCE):
- The following targets have inconsistent incoming links (naming-based asymmetries):
{asym_block}

INTERPRETATION:
- For each target passage X, some source passages have STATE TOKENS that match X well
  ("probable correct sources"), while others have fewer or different tokens and are
  likely incorrect ("suspect sources").
- This usually means that routes with different history (e.g., different companions
  or items) incorrectly converge into a single passage that assumes the wrong state.

TASK:
- For each listed target:
  - ensure that only compatible states converge into a given passage;
  - redirect suspect sources to better matching passages,
    OR create new variants of the target passage with names and text that match
    their distinct state tokens.
- Do NOT rely on variables or conditionals; enforce state consistency purely by
  passage naming and link structure.

OUTPUT:
- ONLY a patch (set of passage edits) in the PATCH FORMAT described above.
""".strip()


# ============================================================
#  PARSING TWEE
# ============================================================

PASSAGE_HEADER_RE = re.compile(r"^::\s*([^\n\[]+)", re.MULTILINE)
LINK_RE = re.compile(r"\[\[(.+?)\]\]")


def parse_passages_and_links(twee_text: str):
    """
    Used for analysis only (links, graph). Works on existing_twee.
    """
    passages = []
    for m in PASSAGE_HEADER_RE.finditer(twee_text):
        name = m.group(1).strip()
        start = m.end()
        passages.append((name, start))

    spans = []
    for i, (name, start) in enumerate(passages):
        end = passages[i + 1][1] if i + 1 < len(passages) else len(twee_text)
        spans.append((name, start, end))

    links = defaultdict(list)
    for name, start, end in spans:
        body = twee_text[start:end]
        for m in LINK_RE.finditer(body):
            inner = m.group(1)
            if "->" in inner:
                target = inner.rsplit("->", 1)[1].strip()
            elif "|" in inner:
                target = inner.rsplit("|", 1)[1].strip()
            else:
                target = inner.strip()
            links[name].append(target)

    return spans, links


def extract_passage_spans(twee_text: str):
    """
    Return detailed spans including header start, body start, end.

    Output: list of tuples
      (name, header_start, body_start, end)
    """
    matches = list(PASSAGE_HEADER_RE.finditer(twee_text))
    spans = []

    if not matches:
        return spans

    for i, m in enumerate(matches):
        name = m.group(1).strip()
        header_start = m.start()
        next_header_start = matches[i + 1].start() if i + 1 < len(matches) else len(twee_text)

        newline_pos = twee_text.find("\n", header_start, next_header_start)
        if newline_pos == -1:
            header_line_end = next_header_start
        else:
            header_line_end = newline_pos

        body_start = header_line_end + 1 if header_line_end < next_header_start else next_header_start
        spans.append((name, header_start, body_start, next_header_start))

    return spans


def strongly_connected_components(graph):
    """
    Tarjan SCC algorithm.
    Returns a list of components, each is a list of nodes.
    """
    index = 0
    indices = {}
    lowlink = {}
    stack = []
    on_stack = set()
    components = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            components.append(comp)

    for v in graph.keys():
        if v not in indices:
            strongconnect(v)

    return components


def find_cycles(graph):
    """
    Find ONLY bad cycles:
    - strongly connected components that actually contain a cycle
    - AND która ma wejście z zewnątrz
    - AND z żadnego węzła nie wychodzi krawędź poza komponent.
    """
    bad_cycles = []
    sccs = strongly_connected_components(graph)

    for comp in sccs:
        comp_set = set(comp)

        if len(comp) == 1:
            n = comp[0]
            if n not in graph.get(n, []):
                continue

        has_incoming_from_outside = False
        has_outgoing_to_outside = False

        for src, targets in graph.items():
            for tgt in targets:
                src_in = src in comp_set
                tgt_in = tgt in comp_set
                if not src_in and tgt_in:
                    has_incoming_from_outside = True
                if src_in and not tgt_in:
                    has_outgoing_to_outside = True

        if has_incoming_from_outside and not has_outgoing_to_outside:
            bad_cycles.append(comp)

    return bad_cycles


def compute_dead_ends(passages, graph):
    endings = [p for p in passages if p.startswith("Ending-")]
    if not endings:
        return endings, sorted(passages)

    rev = defaultdict(list)
    for src, tgts in graph.items():
        for t in tgts:
            rev[t].append(src)

    can_reach = set(endings)
    q = deque(endings)

    while q:
        n = q.popleft()
        for p in rev.get(n, []):
            if p not in can_reach:
                can_reach.add(p)
                q.append(p)

    dead = sorted(p for p in passages if p not in can_reach)
    return endings, dead


def _tokens_from_name(name: str):
    parts = [part.strip() for part in name.split("-") if part.strip()]
    state_parts = parts[1:] if len(parts) > 1 else []
    return set(state_parts)


def detect_symmetries(passages, graph):
    reverse_graph = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            reverse_graph[tgt].append(src)

    symmetries = []

    for target, sources in reverse_graph.items():
        if len(sources) < 2:
            continue

        target_tokens = _tokens_from_name(target)
        if not target_tokens:
            continue

        common_sets = []
        valid = True
        for src in sources:
            src_tokens = _tokens_from_name(src)
            common = target_tokens & src_tokens
            if not common:
                valid = False
                break
            common_sets.append(frozenset(common))

        if not valid:
            continue

        unique_common = set(common_sets)
        if len(unique_common) != 1:
            continue

        symmetries.append((target, sorted(sources)))

    return symmetries


def detect_asymmetries(passages, graph):
    reverse_graph = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            reverse_graph[tgt].append(src)

    asymmetries = []

    for target, sources in reverse_graph.items():
        if len(sources) < 2:
            continue

        target_tokens = _tokens_from_name(target)
        if not target_tokens:
            continue

        common_map = {}
        for src in sources:
            src_tokens = _tokens_from_name(src)
            common = target_tokens & src_tokens
            common_map[src] = (common, len(common))

        common_sets = {frozenset(v[0]) for v in common_map.values()}
        if len(common_sets) <= 1:
            continue

        max_common_size = max(count for (_, count) in common_map.values())
        if max_common_size == 0:
            continue

        probable = []
        suspect = []
        for src, (common, cnt) in common_map.items():
            if cnt == max_common_size and cnt > 0:
                probable.append(src)
            else:
                suspect.append(src)

        if probable and suspect:
            asymmetries.append((target, sorted(probable), sorted(suspect)))

    return asymmetries


def analyze_twee(text: str):
    spans, links = parse_passages_and_links(text)
    passages = set(n for n, _, _ in spans)

    link_targets = []
    for tlist in links.values():
        link_targets.extend(tlist)

    undefined = sorted({t for t in link_targets if t not in passages})

    graph = {p: [] for p in passages}
    for p in passages:
        graph[p] = [t for t in links.get(p, []) if t in passages]

    cycles = find_cycles(graph)
    endings, dead_ends = compute_dead_ends(passages, graph)
    asymmetries = detect_asymmetries(passages, graph)

    return passages, link_targets, undefined, graph, endings, cycles, dead_ends, asymmetries


# ============================================================
#  APPLYING STRUCTURAL PATCHES
# ============================================================

def apply_structural_patch(current_twee: str, patch_twee: str) -> str:
    """
    Apply a structural patch to the current Twee story.

    Patch semantics:
    - For each passage in patch:
      * If its body (after header) is EMPTY/whitespace -> DELETE that passage.
      * Otherwise -> REPLACE or ADD that passage with the patch version.
    """
    orig_spans = extract_passage_spans(current_twee)

    orig_by_name = {}
    orig_order = []
    for name, h_start, b_start, end in orig_spans:
        orig_by_name[name] = (h_start, b_start, end)
        orig_order.append(name)

    patch_spans = extract_passage_spans(patch_twee)

    deleted = set()
    replacements = {}

    for name, h_start, b_start, end in patch_spans:
        body = patch_twee[b_start:end]
        if body.strip() == "":
            deleted.add(name)
        else:
            passage_text = patch_twee[h_start:end].rstrip("\n")
            passage_text = passage_text + "\n"
            replacements[name] = passage_text

    new_parts = []

    # 1) existing passages (preserve order)
    for name in orig_order:
        if name in deleted:
            continue
        if name in replacements:
            new_parts.append(replacements[name])
            del replacements[name]
        else:
            h_start, b_start, end = orig_by_name[name]
            new_parts.append(current_twee[h_start:end])

    # 2) any remaining replacements are NEW passages (names not in orig_order)
    for name, passage_text in replacements.items():
        if new_parts:
            last = new_parts[-1]
            if not last.endswith("\n\n"):
                if not last.endswith("\n"):
                    new_parts[-1] = last + "\n"
                new_parts[-1] = new_parts[-1] + "\n"
        new_parts.append(passage_text)

    return "".join(new_parts)


# ============================================================
#  FIX REQUESTS
# ============================================================

def request(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        r = requests.post(url, json=payload)
    except Exception as e:
        print(f"ERROR contacting Ollama: {e}", file=sys.stderr)
        return ""

    if r.status_code != 200:
        print(f"LLM ERROR {r.status_code}: {r.text}", file=sys.stderr)
        return ""

    try:
        data = r.json()
    except Exception:
        print("Invalid JSON returned.", file=sys.stderr)
        return ""

    return data.get("response", "")


def request_missing_passages(model, description, current, missing, min_passages, print_prompt_only):
    prompt = build_fix_prompt(description, current, missing, min_passages)
    if print_prompt_only:
        print(prompt)
        sys.exit()
    print("\n--- Requesting missing passages...", file=sys.stderr)
    return request(model, prompt)


def request_cycles_fix(model, description, current, cycles, min_passages, print_prompt_only):
    prompt = build_cycles_fix_prompt(description, current, cycles, min_passages)
    if print_prompt_only:
        print(prompt)
        sys.exit()
    print("\n--- Requesting CYCLES structural patch...", file=sys.stderr)
    return request(model, prompt)


def request_dead_fix(model, description, current, dead, min_passages, print_prompt_only):
    prompt = build_dead_fix_prompt(description, current, dead, min_passages)
    if print_prompt_only:
        print(prompt)
        sys.exit()
    print("\n--- Requesting DEAD-ENDS structural patch...", file=sys.stderr)
    return request(model, prompt)


def request_asym_fix(model, description, current, asymmetries, min_passages, print_prompt_only):
    prompt = build_asym_fix_prompt(description, current, asymmetries, min_passages)
    if print_prompt_only:
        print(prompt)
        sys.exit()
    print("\n--- Requesting ASYMMETRY structural patch...", file=sys.stderr)
    return request(model, prompt)


# ============================================================
#  REPORTING
# ============================================================

def print_report(passages,
                 link_targets,
                 undefined,
                 endings,
                 cycles,
                 dead,
                 asymmetries,
                 graph,
                 min_passages,
                 rnd):
    print(f"\n--- REPORT (round {rnd}) ---", file=sys.stderr)
    print(f"Passages: {len(passages)} (min {min_passages})", file=sys.stderr)
    print(f"Links found: {len(link_targets)}", file=sys.stderr)

    if undefined:
        print("Missing passages:", file=sys.stderr)
        for u in undefined:
            print("  -", u, file=sys.stderr)
    else:
        print("No missing passages.", file=sys.stderr)

    print(f"Endings: {len(endings)}", file=sys.stderr)
    for e in endings:
        print("  -", e, file=sys.stderr)

    if cycles:
        print("Cycles detected:", file=sys.stderr)
        for c in cycles:
            print("  -", " -> ".join(c + [c[0]]), file=sys.stderr)
    else:
        print("No cycles.", file=sys.stderr)

    if dead:
        print("Dead-end passages (non-ending):", file=sys.stderr)
        for d in dead:
            print("  -", d, file=sys.stderr)
    else:
        print("No dead-end passages.", file=sys.stderr)

    symmetries = detect_symmetries(passages, graph)
    if symmetries:
        print("\nSymmetries (targets with multiple similarly-named incoming passages):",
              file=sys.stderr)
        for target, sources in symmetries:
            print(f"  {target} (linked from {', '.join(sources)})", file=sys.stderr)
    else:
        print("\nNo naming-based symmetries detected.", file=sys.stderr)

    if asymmetries:
        print("\nAsymmetries (targets with inconsistent incoming naming; probable vs suspect links):",
              file=sys.stderr)
        for target, probable, suspect in asymmetries:
            print(f"  {target} (probable from {', '.join(probable)}; "
                  f"suspect from {', '.join(suspect)})",
                  file=sys.stderr)
    else:
        print("\nNo naming-based asymmetries detected.", file=sys.stderr)


# ============================================================
#  MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Generate or repair a Twine/Twee story using an LLM."
    )
    p.add_argument("model")
    p.add_argument("filename")
    p.add_argument("-o", "--output", default="output.twee")
    p.add_argument("--min-passages", type=int, default=10)
    p.add_argument("--max-fix-rounds", type=int, default=3)
    p.add_argument("--print-prompt-only", action="store_true")
    p.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        help="Skip initial generation and load existing output file into current story."
    )

    args = p.parse_args()

    # Load description
    try:
        description = open(args.filename, "r", encoding="utf8").read()
    except Exception as e:
        print(f"Cannot read input description: {e}", file=sys.stderr)
        sys.exit(1)

    # Option: CONTINUE mode (skip initial generation)
    if args.continue_mode:
        print("\n--- CONTINUE MODE: Loading existing story ---", file=sys.stderr)
        try:
            current = open(args.output, "r", encoding="utf8").read()
        except Exception as e:
            print(f"Cannot read output file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # --- INITIAL GENERATION ---
        prompt = build_prompt(description, args.min_passages)

        if args.print_prompt_only:
            print(prompt)
            return

        print("\n--- Generating initial story ---", file=sys.stderr)
        out_chunks = []

        try:
            with open(args.output, "w", encoding="utf8") as outf:
                r = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": args.model, "prompt": prompt, "stream": True},
                    stream=True,
                )
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    if "response" in data:
                        chunk = data["response"]
                        outf.write(chunk)
                        out_chunks.append(chunk)

                    if data.get("done"):
                        break
        except Exception as e:
            print(f"Generation failed: {e}", file=sys.stderr)
            sys.exit(1)

        current = "".join(out_chunks)

    # === FIX LOOP ====================================================
    for rnd in range(1, args.max_fix_rounds + 1):
        (passages,
         links,
         undefined,
         graph,
         endings,
         cycles,
         dead,
         asymmetries) = analyze_twee(current)

        print_report(
            passages,
            links,
            undefined,
            endings,
            cycles,
            dead,
            asymmetries,
            graph,
            args.min_passages,
            rnd,
        )

        # Exit if everything is OK
        if not undefined and not cycles and not dead and not asymmetries:
            print("\n--- Story OK, no issues left ---", file=sys.stderr)
            break

        # Too many rounds? Stop.
        if rnd == args.max_fix_rounds:
            print("\n--- Max fix rounds reached ---", file=sys.stderr)
            break

        # 1) Fix: Missing passages (local patch)
        if undefined:
            fix = request_missing_passages(
                args.model,
                description,
                current,
                undefined,
                args.min_passages,
                args.print_prompt_only,
            )
            if not fix.strip():
                print("Missing-passages fix returned nothing.", file=sys.stderr)
                break

            fix = "\n" + fix.lstrip("\n")
            current += fix

            try:
                with open(args.output, "a", encoding="utf8") as outf:
                    outf.write(fix)
            except Exception as e:
                print(f"Error writing missing passages: {e}", file=sys.stderr)
                break

            continue

        # 2) Structural fixes w rozbiciu: najpierw cykle, potem dead-endy, potem asymetrie
        patch = ""
        if cycles:
            patch = request_cycles_fix(
                args.model,
                description,
                current,
                cycles,
                args.min_passages,
                args.print_prompt_only,
            )
        elif dead:
            patch = request_dead_fix(
                args.model,
                description,
                current,
                dead,
                args.min_passages,
                args.print_prompt_only,
            )
        elif asymmetries:
            patch = request_asym_fix(
                args.model,
                description,
                current,
                asymmetries,
                args.min_passages,
                args.print_prompt_only,
            )

        if not patch or not patch.strip():
            print("Structural patch returned nothing.", file=sys.stderr)
            break

        new_current = apply_structural_patch(current, patch)
        current = new_current

        try:
            with open(args.output, "w", encoding="utf8") as outf:
                outf.write(current)
        except Exception as e:
            print(f"Error writing structural patch: {e}", file=sys.stderr)
            break

    # Final report
    (passages,
     links,
     undefined,
     graph,
     endings,
     cycles,
     dead,
     asymmetries) = analyze_twee(current)

    print_report(
        passages,
        links,
        undefined,
        endings,
        cycles,
        dead,
        asymmetries,
        graph,
        args.min_passages,
        "FINAL",
    )


if __name__ == "__main__":
    main()
