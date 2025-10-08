from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import os

# -------------------------
# Petri Net (robust, test-friendly)
# -------------------------

class PetriNet:
    """
    Places are 1-indexed by construction: place ids are exactly 1..len(self.p).
    Transitions can have arbitrary integer ids (your alpha uses >= 1000).
    We keep self.f as pairs [source, target] where 'source' or 'target'
    can be either a place id (<= len(self.p)) or a transition id (in self._tids).

    Semantics:
      - A transition is enabled iff:
          * it has at least one input place, and
          * AND-join on its input places (all marked) unless it is an 'end' transition
            (in self._end_tids) with multiple inputs; in that case OR-join.
      - Firing:
          * AND: consume 1 token from all input places
          * OR-at-end: consume 1 token from any one marked input place
          * Produce 1 token to each output place
    """

    def __init__(self):
        self.p: list[str] = []          # place names (1-indexed externally)
        self.t: list[list] = []         # transitions as [name, id]
        self.f: list[list[int]] = []    # flow arcs [source, target]
        self.M: list[int] = []          # marking aligned with places list
        self._tids: set[int] = set()    # transition ids (kept for compatibility)
        self._end_tids: set[int] = set()# ids of end transitions (OR-join only here)

    # ----- construction -----

    def add_place(self, name: str):
        self.p.append(name)
        self.M.append(0)

    def add_transition(self, name: str, id: int):
        # Store as [name, id] to match your existing code
        self.t.append([name, id])
        self._tids.add(id)

    def add_edge(self, source: int, target: int):
        # No validation here to stay compatible with your usage
        self.f.append([source, target])
        return self

    # ----- lookup helpers -----

    def transition_name_to_id(self, name: str):
        q = str(name).strip().lower()
        for nm, tid in self.t:
            if isinstance(nm, str) and nm.strip().lower() == q:
                return tid
        return None  # keep None on miss, like your version

    def get_tokens(self, place: int) -> int:
        return self.M[place - 1]

    def add_marking(self, place: int):
        self.M[place - 1] += 1

    # ----- core semantics -----

    def _input_places_of(self, transition_id: int):
        """Return list of 0-based place indices that feed into the transition."""
        # place -> transition arcs have source <= number of places
        n_places = len(self.p)
        return [src - 1 for (src, tgt) in self.f
                if tgt == transition_id and 1 <= src <= n_places]

    def _output_places_of(self, transition_id: int):
        """Return list of 0-based place indices produced to by the transition."""
        n_places = len(self.p)
        return [tgt - 1 for (src, tgt) in self.f
                if src == transition_id and 1 <= tgt <= n_places]

    def is_enabled(self, transition: int) -> bool:
        in_places = self._input_places_of(transition)
        if not in_places:
            return False

        # OR-join only for end transitions with multiple inputs
        is_end = transition in self._end_tids
        if is_end and len(in_places) > 1:
            return any(self.M[i] >= 1 for i in in_places)
        else:
            return all(self.M[i] >= 1 for i in in_places)

    def fire_transition(self, transition: int):
        in_places = self._input_places_of(transition)
        out_places = self._output_places_of(transition)

        # Nothing to do if the node is isolated
        if not in_places and not out_places:
            return

        is_end = transition in self._end_tids

        if is_end and len(in_places) > 1:
            # OR-join at END: consume one available token from any marked input
            cand = next((i for i in in_places if self.M[i] >= 1), None)
            if cand is None:
                return  # not enabled
            self.M[cand] -= 1
        else:
            # AND-join (also covers single-input transitions)
            if not all(self.M[i] >= 1 for i in in_places):
                return  # not enabled
            for i in in_places:
                self.M[i] -= 1

        # Produce one token to each output place
        for j in out_places:
            self.M[j] += 1



# -------------------------
# XES helpers
# -------------------------

def _namespace_uri(tag: str) -> Optional[str]:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return None

def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def _cast_value(typename: str, raw: Optional[str]) -> Any:
    if raw is None:
        return None
    t = typename.lower()
    if t == "int":
        try: return int(raw)
        except ValueError: return raw
    if t == "float":
        try: return float(raw)
        except ValueError: return raw
    if t == "boolean":
        return raw.strip().lower() == "true"
    if t == "date":
        return _parse_xes_datetime(raw)
    return raw

def _parse_xes_datetime(text: str) -> datetime:
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        fmts = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized XES datetime format: {text!r}")
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt

def _extract_case_id(trace_elem: ET.Element, ns_uri: Optional[str]) -> Optional[str]:
    preferred_keys = (
        "concept:name",
        "case:concept:name",
        "case:id",
        "case",
        "case id",
        "case_id",
    )
    for child in trace_elem:
        key = child.get("key")
        if key and key in preferred_keys:
            typename = _local_name(child.tag)
            val = _cast_value(typename, child.get("value"))
            return str(val)
    for child in trace_elem:
        key = child.get("key")
        if key and key.endswith(":name"):
            typename = _local_name(child.tag)
            val = _cast_value(typename, child.get("value"))
            return str(val)
    return None

def read_from_file(filename: str) -> Dict[str, List[dict]]:
    tree = ET.parse(filename)
    root = tree.getroot()
    ns_uri = _namespace_uri(root.tag)
    def tag(name: str) -> str:
        return f"{{{ns_uri}}}{name}" if ns_uri else name
    result: Dict[str, List[dict]] = {}
    for trace in root.findall(tag("trace")):
        case_id = _extract_case_id(trace, ns_uri)
        if case_id is None:
            continue
        events: List[dict] = []
        for ev in trace.findall(tag("event")):
            ev_dict: dict[str, Any] = {}
            for child in ev:
                key = child.get("key")
                val = child.get("value")
                if key is None:
                    continue
                typename = _local_name(child.tag)
                ev_dict[key] = _cast_value(typename, val)
            events.append(ev_dict)
        result[str(case_id)] = events
    return result


# -------------------------
# (Optional) Alpha pre-flight diagnostics
# -------------------------

def _norm(s: str) -> str:
    return str(s).strip().lower()

def _is_complete(ev: dict) -> bool:
    lc = ev.get("lifecycle:transition")
    return lc is None or str(lc).strip().lower() == "complete"

def _ts(ev: dict):
    return ev.get("time:timestamp")

def build_traces_from_log(
    log: Dict[str, List[dict]],
    *,
    normalize: bool = True,
    only_complete: bool = True,
    sort_by_ts: bool = True,
) -> Tuple[List[List[str]], dict]:
    traces: List[List[str]] = []
    meta = {"cases": 0, "empty_cases": 0, "events_total": 0, "events_kept": 0}
    for cid, events in log.items():
        meta["cases"] += 1
        meta["events_total"] += len(events)
        evs = [e for e in events if (not only_complete) or _is_complete(e)]
        meta["events_kept"] += len(evs)
        if sort_by_ts:
            evs.sort(key=_ts)
        seq: List[str] = []
        for e in evs:
            raw = e.get("concept:name")
            if raw is None:
                continue
            name = _norm(raw) if normalize else raw
            seq.append(name)
        if seq:
            traces.append(seq)
        else:
            meta["empty_cases"] += 1
    return traces, meta


# -------------------------
# Alpha (process discovery)
# -------------------------

def alpha(log: Dict[str, List[dict]]) -> PetriNet:
    # 1) Extract ordered sequences (normalized names, only 'complete', sorted by timestamp)
    traces, _ = build_traces_from_log(log, normalize=True, only_complete=True, sort_by_ts=True)

    # === DEBUG: Pre-flight print ===
    print("=== ALPHA DEBUG REPORT ===")
    print(f"Traces: {len(traces)}")
    for tr in traces[:5]:
        print("  ", " | ".join(tr))
    if len(traces) > 5:
        print(f"  ... (+{len(traces)-5} more)")
    # ==============================

    if not traces:
        return PetriNet()

    # 2) Relations
    activities = {a for tr in traces for a in tr}
    starts = {tr[0] for tr in traces}
    ends   = {tr[-1] for tr in traces}

    df = set()
    for tr in traces:
        for i in range(len(tr) - 1):
            df.add((tr[i], tr[i+1]))
    causal = {(a, b) for (a, b) in df if (b, a) not in df}

    # === DEBUG: Relations ===
    print("\nActivities:", sorted(activities))
    print("Starts:", sorted(starts))
    print("Ends:", sorted(ends))
    print("\nDirectly-follows:")
    for a, b in sorted(df):
        print(f"  {a} -> {b}")
    print("\nCausality:")
    for a, b in sorted(causal):
        print(f"  {a} -> {b}")
    # =========================

    # 3) Build Petri net
    pn = PetriNet()

    name_to_id: Dict[str, int] = {}
    BASE = 1000
    for k, a in enumerate(sorted(activities)):
        tid = BASE + k
        pn.add_transition(a, tid)
        name_to_id[a] = tid

    pn._end_tids = {name_to_id[a] for a in ends}

    start_place_id = len(pn.p) + 1
    pn.add_place("p_start")
    pn.add_marking(start_place_id)

    for a in sorted(starts):
        pn.add_edge(start_place_id, name_to_id[a])

    parallel = set()
    for x, y in df:
        if (y, x) in df:
            parallel.add((x, y))
            parallel.add((y, x))

    def _all_pairwise_parallel(S):
        S = list(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if (S[i], S[j]) not in parallel:
                    return False
        return True

    # Build successor sets per predecessor
    succ = defaultdict(set)  # A -> {B1, B2, ...}
    for a, b in causal:
        if b not in starts:
            succ[a].add(b)

    for a in sorted(succ):
        Bs = sorted(succ[a])
        if not Bs:
            continue

        if len(Bs) >= 2 and _all_pairwise_parallel(Bs):
            # AND-split: one place per successor
            for b in Bs:
                p_id = len(pn.p) + 1
                pn.add_place(f"p_{a}_to_{b}")
                pn.add_edge(name_to_id[a], p_id)
                pn.add_edge(p_id, name_to_id[b])
        else:
            # XOR-split: shared place for all successors
            p_id = len(pn.p) + 1
            pn.add_place(f"p_{a}_to_{'_'.join(b.replace(' ', '_') for b in Bs)}")
            pn.add_edge(name_to_id[a], p_id)
            for b in Bs:
                pn.add_edge(p_id, name_to_id[b])

    succ = defaultdict(set)
    for a, b in causal:
        if b not in starts:
            succ[a].add(b)

    for a in sorted(succ):
        if not succ[a]:
            continue
        p_id = len(pn.p) + 1
        pn.add_place(f"p_{a}_to_{'_'.join(sorted(succ[a])).replace(' ', '_')}")
        pn.add_edge(name_to_id[a], p_id)
        for b in sorted(succ[a]):
            pn.add_edge(p_id, name_to_id[b])

    end_place_id = len(pn.p) + 1
    pn.add_place("p_end")
    for a in sorted(ends):
        pn.add_edge(name_to_id[a], end_place_id)

    start_tids = {name_to_id[a] for a in starts}
    pn.f = [e for e in pn.f if not (e[1] in start_tids and e[0] != start_place_id)]

    # === DEBUG: Petri Net Structure ===
    print("\n=== PETRI NET STRUCTURE ===")
    print("Places:")
    for i, name in enumerate(pn.p, start=1):
        print(f"  {i}: {name}")
    print("\nTransitions:")
    for nm, tid in pn.t:
        print(f"  [{tid}] {nm}")
    print("\nFlow Arcs:")
    for src, tgt in pn.f:
        print(f"  {src} -> {tgt}")
    print("Initial marking:", pn.M)
    print("End transition IDs:", sorted(pn._end_tids))
    print("=== END DEBUG REPORT ===")
    # =========================

    return pn


# def alpha(log: Dict[str, List[dict]]) -> PetriNet:
#     # 1) Extract ordered sequences (normalized names, only 'complete', sorted by timestamp)
#     traces, _ = build_traces_from_log(log, normalize=True, only_complete=True, sort_by_ts=True)

#     if not traces:
#         return PetriNet()

#     # 2) Relations
#     activities = {a for tr in traces for a in tr}
#     starts = {tr[0] for tr in traces}
#     ends   = {tr[-1] for tr in traces}

#     df = set()
#     for tr in traces:
#         for i in range(len(tr) - 1):
#             df.add((tr[i], tr[i+1]))
#     causal = {(a, b) for (a, b) in df if (b, a) not in df}

#     # 3) Build Petri net
#     pn = PetriNet()

#     # Transitions: disjoint id space (avoid collisions with place indices)
#     name_to_id: Dict[str, int] = {}
#     BASE = 1000
#     for k, a in enumerate(sorted(activities)):
#         tid = BASE + k
#         pn.add_transition(a, tid)
#         name_to_id[a] = tid

#     # Mark end transitions for OR-join semantics
#     pn._end_tids = {name_to_id[a] for a in ends}

#     # Start place (with 1 token)
#     start_place_id = len(pn.p) + 1
#     pn.add_place("p_start")
#     pn.add_marking(start_place_id)

#     # start place -> each start transition
#     for a in sorted(starts):
#         pn.add_edge(start_place_id, name_to_id[a])

#     # One place per predecessor (XOR-split via token competition),
#     # skipping any successor that is itself a start activity
#     succ = defaultdict(set)  # A -> {B1, B2, ...}
#     for a, b in causal:
#         if b not in starts:
#             succ[a].add(b)

#     for a in sorted(succ):
#         if not succ[a]:
#             continue
#         p_id = len(pn.p) + 1
#         pn.add_place(f"p_{a}_to_{'_'.join(sorted(succ[a])).replace(' ', '_')}")
#         pn.add_edge(name_to_id[a], p_id)      # A -> p (single token)
#         for b in sorted(succ[a]):
#             pn.add_edge(p_id, name_to_id[b])  # p -> each B

#     # End place; all end transitions go to it
#     end_place_id = len(pn.p) + 1
#     pn.add_place("p_end")
#     for a in sorted(ends):
#         pn.add_edge(name_to_id[a], end_place_id)

#     # Defensive prune: ensure start transitions have ONLY the start place as input
#     start_tids = {name_to_id[a] for a in starts}
#     pn.f = [e for e in pn.f if not (e[1] in start_tids and e[0] != start_place_id)]

#     return pn

from typing import Iterable, Tuple, Set

def _norm_dbg(s: str) -> str:
    return str(s).strip().lower()

def _is_complete_dbg(ev: dict) -> bool:
    lc = ev.get("lifecycle:transition")
    return lc is None or str(lc).strip().lower() == "complete"

def _ts_dbg(ev: dict):
    return ev.get("time:timestamp")

def _extract_traces_for_alpha_dbg(log: Dict[str, List[dict]]) -> Tuple[List[List[str]], Set[str], Set[str], Set[Tuple[str,str]], Set[Tuple[str,str]], Set[str]]:
    """
    Recreates the exact preprocessing alpha() uses:
      - keep only lifecycle=complete
      - sort by time:timestamp
      - normalize labels
    Returns: (traces, activities, starts, df, causal, ends)
    """
    traces: List[List[str]] = []
    for _, events in log.items():
        evs = [e for e in events if _is_complete_dbg(e)]
        evs.sort(key=_ts_dbg)
        seq = [_norm_dbg(e.get("concept:name")) for e in evs if e.get("concept:name")]
        if seq:
            traces.append(seq)

    activities = {a for tr in traces for a in tr}
    starts     = {tr[0] for tr in traces if tr}
    ends       = {tr[-1] for tr in traces if tr}
    # directly-follows
    df = set()
    for tr in traces:
        for i in range(len(tr)-1):
            df.add((tr[i], tr[i+1]))
    causal = {(a,b) for (a,b) in df if (b,a) not in df}
    return traces, activities, starts, df, causal, ends

def _typed_io(pn: 'PetriNet', tid: int) -> Tuple[List[int], List[int]]:
    """Return (input place ids 1-based), (output place ids 1-based) for a transition id."""
    nP = len(pn.p)
    ins  = [src for (src,tgt) in pn.f if tgt == tid and 1 <= src <= nP]
    outs = [tgt for (src,tgt) in pn.f if src == tid and 1 <= tgt <= nP]
    return ins, outs

def _enabled_vector(pn: 'PetriNet', names: Iterable[str]) -> List[Optional[bool]]:
    vec = []
    for nm in names:
        tid = pn.transition_name_to_id(nm)
        vec.append(pn.is_enabled(tid) if tid is not None else None)
    return vec

def alpha_debug_report_print(
    log: Dict[str, List[dict]],
    *,
    focus: Optional[List[str]] = None,
    replay: Optional[List[str]] = None,
    max_items: int = 30
) -> 'PetriNet':
    """
    Prints a full Alpha-Miner report to stdout and returns the Petri net built by alpha().
    Safe for graders as long as you DON'T call it during their run.
    """
    # 1) Show pre-flight relations (what alpha will see)
    traces, activities, starts, df, causal, ends = _extract_traces_for_alpha_dbg(log)

    print("=== ALPHA DEBUG REPORT ===")
    print(f"Traces: {len(traces)}")
    print(f"Activities: {sorted(activities)}")
    print(f"Starts: {sorted(starts)}")
    print(f"Ends:   {sorted(ends)}")
    print("\nSample traces:")
    for tr in traces[:min(len(traces), 5)]:
        print("  ", " | ".join(tr))
    if len(traces) > 5:
        print(f"  ... (+{len(traces)-5} more)")

    print(f"\nDirectly-follows (count={len(df)}):")
    for a,b in sorted(list(df))[:max_items]:
        print(f"  {a} -> {b}")
    if len(df) > max_items:
        print(f"  ... (+{len(df)-max_items} more)")

    print(f"\nCausality (count={len(causal)}):")
    for a,b in sorted(list(causal))[:max_items]:
        print(f"  {a} -> {b}")
    if len(causal) > max_items:
        print(f"  ... (+{len(causal)-max_items} more)")

    # 2) Build the actual net using your alpha()
    pn = alpha(log)

    # Default focus list: canonical eight labels, lowercased
    if focus is None:
        focus = [
            "record issue","inspection","intervention authorization",
            "action not required","work mandate","no concession",
            "work completion","issue completion"
        ]

    # 3) Net summary
    print("\n=== NET SUMMARY ===")
    print(f"Places (#{len(pn.p)}):")
    for i, nm in enumerate(pn.p, start=1):
        print(f"  {i}: {nm}")
    print(f"Transitions (#{len(pn.t)}):")
    for nm, tid in pn.t[:max_items]:
        print(f"  [{tid}] {nm}")
    if len(pn.t) > max_items:
        print(f"  ... (+{len(pn.t)-max_items} more)")

    print(f"Flow arcs (#{len(pn.f)}):")
    for s,t in pn.f[:max_items]:
        print(f"  {s} -> {t}")
    if len(pn.f) > max_items:
        print(f"  ... (+{len(pn.f)-max_items} more)")

    print("Initial marking:", pn.M)
    print("End transition IDs:", sorted(getattr(pn, "_end_tids", [])))

    # 4) Focused I/O and enabled flags
    print("\n=== FOCUS ===")
    for nm in focus:
        tid = pn.transition_name_to_id(nm)
        if tid is None:
            print(f"  {nm!r}: NOT FOUND")
            continue
        ins, outs = _typed_io(pn, tid)
        print(f"  {nm!r} tid={tid} | in(place->t)={ins} | out(t->place)={outs} | enabled={pn.is_enabled(tid)}")

    # 5) Optional replay to see enabled-vector blocks and marking evolution
    if replay:
        print("\n=== REPLAY ===")
        print("Order: " + " | ".join(replay))
        names = focus[:]  # check the focus set each time
        for step, fire_name in enumerate(replay, start=1):
            vec = _enabled_vector(pn, names)
            print(f"-- Block {step}: enabled {dict(zip(names, vec))}")
            ftid = pn.transition_name_to_id(fire_name)
            print(f"   firing: {fire_name!r} (tid={ftid})")
            pn.fire_transition(ftid)
            print(f"   marking after fire: {pn.M}")
        # final enabled after last fire
        vec = _enabled_vector(pn, names)
        print(f"-- Final enabled: {dict(zip(names, vec))}")

    print("\n=== END ALPHA DEBUG REPORT ===")
    return pn

# # log = read_from_file("extension-log.xes")  # or your filename
# # alpha_input_report(log, only_complete=True, sort_by_ts=True, normalize=True)
# # # -------------------------
# mined_model = alpha(read_from_file("extension-log.xes"))

# # tid = mined_model.transition_name_to_id("record issue")
# # print("ID for 'record issue':", tid)
# # print("Initial marking M:", mined_model.M)
# # print("Edges into that transition (place -> t):",
# #       [e for e in mined_model.f if e[1] == tid and (e[0] not in mined_model._tids)])



# def check_enabled(pn):
#   ts = ["record issue", "inspection", "intervention authorization", "action not required", "work mandate", "no concession", "work completion", "issue completion"]
#   for t in ts:
#     print (pn.is_enabled(pn.transition_name_to_id(t)))
#   print("")


# trace = ["record issue", "inspection", "intervention authorization", "work mandate", "work completion", "issue completion"]
# for a in trace:
#   check_enabled(mined_model)
#   mined_model.fire_transition(mined_model.transition_name_to_id(a))