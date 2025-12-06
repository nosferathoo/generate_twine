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


def build_structural_fix_prompt(description: str,
                                existing_twee: str,
                                cycles,
                                dead_ends,
                                min_passages: int) -> str:
    """
    Prompt asking the LLM to fix structural problems (cycles, dead ends)
    by rewriting the whole story into a new, corrected Twee.
    Now also includes a list of detected asymmetries (suspect source->target links).
    """
    common = build_common_instructions(min_passages)

    # Format cycles
    cycles_lines = []
    for i, cycle in enumerate(cycles, 1):
        arrow = " -> ".join(cycle + [cycle[0]]) if cycle else ""
        cycles_lines.append(f"- Cycle {i}: {arrow}")
    cycles_block = "\n".join(cycles_lines) if cycles_lines else "None"

    # Format dead-end passages
    dead_block = "\n".join(f"- {name}" for name in dead_ends) if dead_ends else "None"

    # --- Wykrywanie asymetrii na podstawie istniejącego Twee ---
    asym_block = "None"
    try:
        spans, links = parse_passages_and_links(existing_twee)
        passages = set(name for name, _, _ in spans)
        graph = {name: links.get(name, []) for name, _, _ in spans}
        asymmetries = detect_asymmetries(passages, graph)

        asym_lines = []
        for target, probable, suspect in asymmetries:
            for src in suspect:
                asym_lines.append(f"- {src} -> {target}")
        if asym_lines:
            asym_block = "\n".join(asym_lines)
    except Exception as e:
        # W razie problemów z analizą wolimy mieć prompt bez tej sekcji
        asym_block = f"ERROR while computing asymmetries: {e}"

    prompt = f"""
{common}

Description:
\"\"\"{description.strip()}\"\"\"

You previously generated the following Twine/Twee story:

\"\"\"{existing_twee}\"\"\"


STRUCTURAL PROBLEMS DETECTED:

1) CYCLES in the graph:
{cycles_block}

2) PASSAGES THAT CANNOT REACH ANY ENDING (dead-end branches):
{dead_block}

3) SUSPECT LINKS (naming-based asymmetries, likely incorrect source->target):
{asym_block}

TASK:
- Rewrite the ENTIRE story as a NEW valid Twee story that:
  - removes or breaks all bad cycles and dead-end-only branches,
  - ensures EVERY passage can reach at least one Ending-,
  - fixes ALL suspect links listed above by enforcing state consistency
- Preserve the world, main ideas and overall tone, but you MAY:
  - restructure passages,
  - rename passages (respecting naming rules),
  - change links and graph structure as needed.
- Obey ALL rules from the common instructions, including:
  - panel-like passages,
  - LOCATION + STATE token naming (no '_' and no meaningless tokens),
  - Ending- passages with no outgoing links,
  - no variables, no macros, no scripting.

OUTPUT:
- ONLY the full corrected Twee story (no comments, no explanation).
"""
    return prompt.strip()


# ============================================================
#  PARSING TWEE
# ============================================================

PASSAGE_HEADER_RE = re.compile(r"^::\s*([^\n\[]+)", re.MULTILINE)
LINK_RE = re.compile(r"\[\[(.+?)\]\]")


def parse_passages_and_links(twee_text: str):
    passages = []
    for m in PASSAGE_HEADER_RE.finditer(twee_text):
        name = m.group(1).strip()
        start = m.end()
        passages.append((name, start))

    spans = []
    for i, (name, start) in enumerate(passages):
        end = passages[i+1][1] if i+1 < len(passages) else len(twee_text)
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
    Innymi słowy: gracz może wejść w pętlę, ale nie może z niej wyjść.
    """
    bad_cycles = []
    sccs = strongly_connected_components(graph)

    # Zbuduj pomocnicze mapy krawędzi dla szybszego sprawdzania
    for comp in sccs:
        comp_set = set(comp)

        # 1) Czy w ogóle jest cykl w tej SCC?
        #    - więcej niż 1 węzeł
        #    - albo self-loop
        if len(comp) == 1:
            n = comp[0]
            if n not in graph.get(n, []):
                # brak self-loop → to nie cykl
                continue

        # 2) Czy można wejść do tej SCC z zewnątrz?
        has_incoming_from_outside = False
        # 3) Czy jest wyjście z SCC na zewnątrz?
        has_outgoing_to_outside = False

        for src, targets in graph.items():
            for tgt in targets:
                src_in = src in comp_set
                tgt_in = tgt in comp_set
                if not src_in and tgt_in:
                    has_incoming_from_outside = True
                if src_in and not tgt_in:
                    has_outgoing_to_outside = True

        # Zła pętla = wejście z zewnątrz jest, wyjścia brak
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
    """
    Rozbij nazwę passage na tokeny po '-', pomijając pierwszy token
    (prawdopodobnie nazwa lokacji), i wyczyść spacje.
    Używane do porównywania 'słów kluczowych' w nazwach.
    Przykład:
      'CityGate-WithLyra-HasKeycard' -> {'WithLyra', 'HasKeycard'}
    """
    parts = [part.strip() for part in name.split("-") if part.strip()]
    # pomijamy pierwszy element (lokacja / bazowa nazwa)
    state_parts = parts[1:] if len(parts) > 1 else []
    return set(state_parts)

def detect_symmetries(passages, graph):
    """
    Symetria dla passage X:
    - istnieje co najmniej 2 źródła linków do X,
    - każde źródło ma NIEPUSTY wspólny zbiór tokenów z X,
    - i TEN SAM zbiór 'common tokens' dla wszystkich źródeł.

    Zwraca listę:
    [
      (target_name, [source1, source2, ...]),
      ...
    ]
    """
    # odwrotny graf: X -> [źródła, które linkują do X]
    reverse_graph = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            reverse_graph[tgt].append(src)

    symmetries = []

    for target, sources in reverse_graph.items():
        # symetria ma sens tylko, jeśli co najmniej 2 źródła
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
                # jakiś link w ogóle nie dzieli tokenów z X → brak symetrii dla X
                valid = False
                break
            common_sets.append(frozenset(common))

        if not valid:
            continue

        # wszystkie common_sets muszą być identyczne
        unique_common = set(common_sets)
        if len(unique_common) != 1:
            # są różne zestawy wspólnych tokenów → brak symetrii wg Twojej definicji
            continue

        # Jeśli doszliśmy tutaj − mamy symetrię: wszystkie źródła "wyglądają podobnie" względem X
        symmetries.append((target, sorted(sources)))

    return symmetries
    
def detect_asymmetries(passages, graph):
    """
    Asymetria dla passage X:
    - istnieje >=2 źródła linków do X,
    - nie wszystkie źródła wyglądają tak samo względem X (różna ilość / zestaw wspólnych tokenów),
    - wyznaczamy:
      * źródła z MAKSYMALNĄ liczbą wspólnych tokenów z X jako "prawdopodobnie poprawne",
      * pozostałe jako "podejrzane" (potencjalnie błędne linki).

    Zwraca listę:
    [
      (target_name, [probable_correct_sources], [suspect_sources]),
      ...
    ]
    """
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

        # policz wspólne tokeny dla każdego źródła
        common_map = {}  # src -> (common_tokens_set, count)
        for src in sources:
            src_tokens = _tokens_from_name(src)
            common = target_tokens & src_tokens
            common_map[src] = (common, len(common))

        # jeśli wszystkie mają ten sam zestaw (albo wszystkie 0) → brak asymetrii
        common_sets = {frozenset(v[0]) for v in common_map.values()}
        if len(common_sets) <= 1:
            continue

        # znajdź maksymalną liczbę wspólnych tokenów
        max_common_size = max(count for (_, count) in common_map.values())
        if max_common_size == 0:
            # nikt nie dzieli żadnych tokenów z X → nic sensownego nie wyłonimy
            continue

        probable = []
        suspect = []
        for src, (common, cnt) in common_map.items():
            if cnt == max_common_size and cnt > 0:
                probable.append(src)
            else:
                suspect.append(src)

        # Asymetria ma sens tylko gdy mamy kogoś "prawdopodobnie poprawnego"
        # i kogoś "podejrzanego"
        if probable and suspect:
            asymmetries.append((target, sorted(probable), sorted(suspect)))

    return asymmetries    
    
def analyze_twee(text: str):
    spans, links = parse_passages_and_links(text)
    passages = set(n for n, _, _ in spans)

    # wszystkie cele linków (do liczenia undefined)
    link_targets = []
    for tlist in links.values():
        link_targets.extend(tlist)

    undefined = sorted({t for t in link_targets if t not in passages})

    # graf: passage -> lista targetów będących istniejącymi passages
    graph = {p: [] for p in passages}
    for p in passages:
        graph[p] = [t for t in links.get(p, []) if t in passages]

    cycles = find_cycles(graph)
    endings, dead_ends = compute_dead_ends(passages, graph)
    asymmetries = detect_asymmetries(passages, graph)

    return passages, link_targets, undefined, graph, endings, cycles, dead_ends, asymmetries


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
    except:
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


def request_structural_fix(model, description, current, cycles, dead, min_passages, print_prompt_only):
    prompt = build_structural_fix_prompt(description, current, cycles, dead, min_passages)
    if print_prompt_only:
        print(prompt)
        sys.exit()        
    print("\n--- Requesting structural fix...", file=sys.stderr)
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
            print("  -", " -> ".join(c+[c[0]]), file=sys.stderr)
    else:
        print("No cycles.", file=sys.stderr)

    if dead:
        print("Dead-end passages (non-ending):", file=sys.stderr)
        for d in dead:
            print("  -", d, file=sys.stderr)
    else:
        print("No dead-end passages.", file=sys.stderr)
        
    # --- Symetrie w linkach (w sensie nazw passage) ---
    symmetries = detect_symmetries(passages, graph)
    if symmetries:
        print("\nSymmetries (targets with multiple similarly-named incoming passages):",
              file=sys.stderr)
        for target, sources in symmetries:
            print(f"  {target} (linked from {', '.join(sources)})", file=sys.stderr)
    else:
        print("\nNo naming-based symmetries detected.", file=sys.stderr)
        
    # --- Asymetrie w linkach (potencjalnie błędne przejścia) ---
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

        # Option: print prompt and exit
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

        # Exit if everything is OK (BRAK: undefined, cycles, dead, asymmetries)
        if not undefined and not cycles and not dead and not asymmetries:
            print("\n--- Story OK, no issues left ---", file=sys.stderr)
            break

        # Too many rounds? Stop.
        if rnd == args.max_fix_rounds:
            print("\n--- Max fix rounds reached ---", file=sys.stderr)
            break

        # 1) Fix: Missing passages (lokalna łatka, bez przepisywania całości)
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

        # 2) Fix: Structural rewrite (pętle, dead-endy LUB asymetrie)
        if cycles or dead or asymmetries:
            fix = request_structural_fix(
                args.model,
                description,
                current,
                cycles,
                dead,
                args.min_passages,
                args.print_prompt_only,
            )
            if not fix.strip():
                print("Structural fix returned nothing.", file=sys.stderr)
                break

            current = fix

            try:
                with open(args.output, "w", encoding="utf8") as outf:
                    outf.write(current)
            except Exception as e:
                print(f"Error writing structural fix: {e}", file=sys.stderr)
                break

            continue

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
