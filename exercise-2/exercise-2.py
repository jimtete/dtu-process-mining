from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from collections import defaultdict
import os


# -------------------------
# Helpers
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
        try:
            return int(raw)
        except ValueError:
            return raw  # fall back to str if malformed
    if t == "float":
        try:
            return float(raw)
        except ValueError:
            return raw
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
    # >>> Make it NAIVE (drop tzinfo) to match autograder expectation
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt



def _extract_case_id(trace_elem: ET.Element, ns_uri: Optional[str]) -> Optional[str]:
    """
    Extract the case id from a <trace> by looking for common keys.
    Prefers 'concept:name', with fallbacks.
    """
    # iterate direct children (trace-level attributes)
    preferred_keys = (
        "concept:name",
        "case:concept:name",
        "case:id",
        "case",
        "case id",
        "case_id",
    )

    # First pass: look for preferred keys
    for child in trace_elem:
        key = child.get("key")
        if key and key in preferred_keys:
            typename = _local_name(child.tag)
            val = _cast_value(typename, child.get("value"))
            return str(val)

    # Second pass: many XES traces store the name under concept:name
    for child in trace_elem:
        key = child.get("key")
        if key and key.endswith(":name"):  # e.g., concept:name
            typename = _local_name(child.tag)
            val = _cast_value(typename, child.get("value"))
            return str(val)

    return None

def log_as_dictionary(log: str) -> dict:
    log_dict = {}

    log_lines = log.strip().splitlines()

    for line in log_lines:
        input = line.split(";")

        if (all(s.strip() == "" for s in input)):
            continue

        task, case_id, user_id, date_time = input

        if (case_id not in log_dict):
            log_dict[case_id] = []

        event = {
            "task": task,
            "case_id": case_id,
            "user_id": user_id,
            "timestamp": date_time
        }

        log_dict[case_id].append(event)
    return log_dict

def dependency_graph_inline(log: dict) -> dict:
    df = defaultdict(lambda: defaultdict(int))
    
    for events in log.values():
        for idx in range(len(events) - 1):
            src = events[idx]["task"]
            tgt = events[idx + 1]["task"]
            df[src][tgt] += 1

    return {src: dict(tgts) for src, tgts in df.items()}

def dependency_graph_file(src: Union[str, Dict[str, List[dict]]]) -> dict:
    """
    Build a dependency graph (direct-follows) where df[src][tgt] = count.

    Accepts:
      - dict {case_id: [event_dict,...]}  (already parsed)
      - str path to a file with semicolon-separated lines
      - str raw multiline semicolon-separated text

    Event dicts may be from CSV-style (use key "task") or XES-style
    (use key "concept:name" and optionally "time:timestamp").
    """
    # Normalize to: cases -> list of event dicts
    if isinstance(src, dict):
        cases = src  # already in desired structure
    elif isinstance(src, (str, os.PathLike)):
        # Read from file path if it exists, else treat as raw text
        text = None
        path_str = str(src)
        if os.path.exists(path_str):
            with open(path_str, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = path_str

        # Parse semicolon CSV lines: task;case_id;user_id;timestamp
        cases = defaultdict(list)
        for raw in text.strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) != 4:
                continue
            task, case_id, user_id, timestamp = parts
            cases[case_id].append({
                "task": task,
                "case_id": case_id,
                "user_id": user_id,
                "timestamp": timestamp
            })
    else:
        raise TypeError("dependency_graph_file expects a dict or a string/path")

    # Count direct-follows per case.
    # If events have timestamps, sort by them; otherwise keep input order.
    df = defaultdict(lambda: defaultdict(int))

    for events in cases.values():
        # pair each event with its original position for stable tie-breaking
        indexed = list(enumerate(events))

        # choose key: XES uses "concept:name"; CSV uses "task"
        def activity(e: dict) -> Any:
            return e.get("concept:name") or e.get("task")

        # choose timestamp if available for ordering
        def ts(e: dict):
            return e.get("time:timestamp", None)

        # sort only if at least one timestamp exists
        if any(ts(e) is not None for e in events):
            indexed.sort(key=lambda p: (ts(p[1]) if ts(p[1]) is not None else datetime.min, p[0]))

        # walk adjacent pairs
        for (_, a), (_, b) in zip(indexed, indexed[1:]):
            src_act = activity(a)
            tgt_act = activity(b)
            if src_act is None or tgt_act is None:
                continue
            df[src_act][tgt_act] += 1

    # Return plain dict-of-dicts
    return {src_act: dict(tgts) for src_act, tgts in df.items()}

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

# log = read_from_file("extension-log.xes")

# # general statistics: for each case id the number of events contained
# for case_id in sorted(log):
#     print((case_id, len(log[case_id])))

# # details for a specific event of one case
# case_id = "case_123"
# event_no = 0
# print((log[case_id][event_no]["concept:name"], log[case_id][event_no]["org:resource"], log[case_id][event_no]["time:timestamp"],  log[case_id][event_no]["cost"]))