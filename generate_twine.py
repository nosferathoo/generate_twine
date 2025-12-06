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
    Bardzo skrócone zasady + dodatkowe twarde zakazy (Alone, [] w endingach).
    """
    return f"""
You are an expert Twine/Twee interactive fiction author.

GLOBAL:
- Output ONLY valid Twee code (no comments, no markdown).
- Story language = description language.
- NO variables, NO macros, NO <<if>>, NO <<set>>, NO scripting.
- Narrative state = ONLY passage names + link graph.
- Passage names MUST NOT contain '_' anywhere.
- Story must have at least {min_passages} passages.

PASSAGE NAMES (LOCATION + STATE TOKENS):
- Form:  LocationName[-Token1-Token2-...]
- LocationName:
  - First chunk (before first '-'), e.g. SkybreakerSublevel2, RelayStationCorridor, CityGateNight.
  - MUST NOT contain '-' or '_'.
  - Describes where/when this panel happens.
- State tokens (after first '-'):
  - Encode important facts only, e.g.:
    - MetLyra, WithLyra, LostLyra
    - HasKeycard, LostKeycard, LeftKeycardBehind
    - JoinedFactionA, JoinedFactionB, BetrayedFactionA
- ABSENCE OF A TOKEN = default:
  - No WithLyra / MetLyra => player never met/is not with Lyra.
  - No HasKeycard => player does not carry the keycard.
- FORBIDDEN / USELESS TOKENS (NEVER USE):
  - Alone, EmptyInventory, Default, Normal, Generic, None, Nothing, Standard
  - Do NOT encode “alone” as a token. If player is alone, just omit WithX/MetX tokens.

PASSAGE = COMIC PANEL:
- One passage = one clear panel:
  - one location (LocationName),
  - one concrete moment,
  - one immediate situation.
- Time flows BETWEEN passages, not inside one long paragraph.
- Do NOT compress several major actions (travel, entering area, finding & taking key item)
  into a single passage. Use 2–3+ passages for important steps.

ITEMS / GOALS:
- Important items/goals must NOT just lie around unguarded in the same passage where they appear.
- Reaching/obtaining a key item should involve:
  - noticing or being told,
  - approaching through danger/obstacles,
  - dealing with guards/traps/locks/moral choice,
  - THEN obtaining or failing to obtain it.

LINKS:
- Allowed format ONLY: [[Text->PassageName]]
- NO spaces around '->'.
- Each link on its own line (one line = one choice).
- All link targets MUST exist as passages.
- OUTSIDE OF LINKS, NEVER USE '[' or ']' in the story text.
  - No decorative [[THE END]], no brackets used as styling.

ENDINGS:
- Ending passages MUST have names starting with:  Ending-
- Ending passages MUST:
  - have NO outgoing links at all,
  - contain NO '[' or ']' characters in the body (plain text ending only).
- Include ≥1 good/satisfying ending and ≥3 bad endings (death/failure/etc.).

CONSISTENCY & BRANCHING:
- No contradictions: text in a passage must match the state encoded in its name and all routes leading there.
- Do NOT mention characters/items that are not guaranteed by state tokens and previous links.
- Symmetric states (same tokens) may converge to shared passages.
- Asymmetry (different/no tokens converging into a passage whose name encodes a specific state)
  is suspicious and should be avoided or fixed.

GRAPH RULES:
- NO endless closed cycles: player must never be trapped in a loop with no exit.
- Backtracking loops allowed only if there is an exit to progress from at least one node.
- EVERY non-ending passage must be able to reach an Ending- eventually.
- No non-ending dead-ends.

VERIFY BEFORE OUTPUT:
- All links valid and formatted correctly.
- No '[' or ']' in non-link text (and none at all in Ending- passages).
- No broken links.
- No non-ending dead-ends.
- No bad cycles with no exit.
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

TASK (MISSING PASSAGES ONLY):
- DO NOT modify or repeat existing passages.
- For EACH missing name output EXACTLY ONE Twee passage.
- DO NOT invent new passage names beyond the missing list.
- Use valid Twee headers and links (respect all global rules).
- Text must match the world + state naming logic.

OUTPUT:
- ONLY the missing passages.
""".strip()


def _patch_instructions(min_passages: int) -> str:
    """
    Wspólny fragment dla wszystkich promptów patchowych.
    """
    common = build_common_instructions(min_passages)
    return f"""
{common}

IMPORTANT: RETURN ONLY A PATCH, NOT THE FULL STORY.

PATCH FORMAT (PASSAGE-LEVEL EDITS):
- To MODIFY an existing passage:
  - Output its FULL header + body, e.g.:
      :: PassageName
      (full body)

- To ADD a new passage:
  - Same: full header + body.

- To DELETE a passage:
  - Output only a single header line:
      :: NameOfDeletedPassage
    with NO body (no text, no blank line, nothing).

- To RENAME a passage:
  - Delete old name:
      :: OldPassageName
    (no body)
  - And output full passage with the new name:
      :: NewPassageName
      (full body)

RULES FOR PATCH:
- DO NOT output any unchanged passages.
- DO NOT output prose explanations, comments, markdown, or code fences.
- Obey all naming, brackets, link and story rules from the common instructions.
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

STRUCTURAL PROBLEM (CYCLES):
- Bad cycles detected (player can enter, but cannot escape to progress/endings):
{cycles_block}

TASK:
- Break or restructure these cycles so that:
  - no closed loop traps the player forever,
  - every cycle gets at least one exit toward progress and eventually an Ending-,
  - or is replaced by an acyclic structure.
- You MAY adjust links, split passages, rename or delete problematic passages.

OUTPUT:
- ONLY a patch in the specified PATCH FORMAT.
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

STRUCTURAL PROBLEM (DEAD-ENDS):
- The following passages CANNOT reach any Ending- and are not endings themselves:
{dead_block}

TASK:
- For every passage above, ensure the player can eventually reach at least one Ending-.
- Non-ending passages must not be terminal dead-ends.
- You MAY add passages, change links, rename or delete passages as needed,
  while keeping the story and state-encoding coherent.

OUTPUT:
- ONLY a patch in the specified PATCH FORMAT.
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
                f"    Suspect sources: {susp_s}"
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
- Targets with inconsistent incoming states (naming-based asymmetries):
{asym_block}

INTERPRETATION:
- For each target, some sources have state tokens matching it well (probable),
  others have fewer/different tokens (suspect) and likely represent different
  narrative states (e.g. with/without companions or items).

TASK:
- For each target:
  - ensure only compatible states converge to one passage,
  - redirect suspect sources to better matching passages,
  - or create state-specific variants (different tokens in name + text).
- Enforce state consistency purely via passage naming and link graph (no variables).

OUTPUT:
- ONLY a patch in the specified PATCH FORMAT.
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
    - strongly connected components that contain a cycle
    - AND have an incoming edge from outside
    - AND have no outgoing edge to outside.
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

    # 2) new passages (names not in orig_order)
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