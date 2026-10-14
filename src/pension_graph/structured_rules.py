from __future__ import annotations

import re
from dataclasses import dataclass

UNIT_PATTERN = r"억원|만원|개월|퍼센트|세|살|년|월|일|회|원|%"
MEASURE_VALUE = re.compile(
    rf"(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})"
)
BOUNDED_INTERVALS = (
    re.compile(
        rf"(?P<low>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})\s*이상\s*"
        rf"(?P<high>\d[\d,]*(?:\.\d+)?)\s*(?P=unit)\s*(?P<upper>미만|이하)"
    ),
    re.compile(
        rf"(?P<low>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})\s*[~～-]\s*"
        rf"(?P<high>\d[\d,]*(?:\.\d+)?)\s*(?P=unit)\s*(?P<upper>미만|이하)?"
    ),
)
OPEN_LOWER = re.compile(
    rf"(?P<low>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})\s*(?P<lower>이상|초과)"
)
OPEN_DASH = re.compile(
    rf"(?P<low>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})\s*[~～-](?!\s*\d)"
)
OPEN_UPPER = re.compile(
    rf"(?P<high>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_PATTERN})\s*(?P<upper>미만|이하)"
)
TABLE_HINT = re.compile(
    r"^table_range_match:(?P<table_id>.+):row:(?P<row_index>\d+) \| (?P<cells>.*)$",
    re.DOTALL,
)
NORMALIZE_TEXT = re.compile(r"[^0-9A-Za-z가-힣%]+")


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _canonical_unit(unit: str) -> str:
    if unit == "퍼센트":
        return "%"
    return "세" if unit == "살" else unit


@dataclass(frozen=True)
class Measure:
    value: float
    unit: str
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class NumericInterval:
    unit: str
    raw: str
    start: int
    end: int
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = False

    def contains(self, measure: Measure) -> bool:
        if self.unit != measure.unit:
            return False
        if self.low is not None and (
            measure.value < self.low
            or (
                measure.value == self.low and not self.low_inclusive
            )
        ):
            return False
        return not (
            self.high is not None
            and (
                measure.value > self.high
                or (measure.value == self.high and not self.high_inclusive)
            )
        )


@dataclass(frozen=True)
class StructuredCell:
    header: str
    value: str


@dataclass(frozen=True)
class StructuredTableRow:
    table_id: str
    row_index: int
    cells: tuple[StructuredCell, ...]


@dataclass(frozen=True)
class TableResolution:
    query_measure: Measure
    condition: StructuredCell
    output: StructuredCell
    target_score: int
    table_id: str
    row_index: int


def extract_measures(text: str) -> list[Measure]:
    return [
        Measure(
            value=_number(match.group("value")),
            unit=_canonical_unit(match.group("unit")),
            raw=match.group(0).strip(),
            start=match.start(),
            end=match.end(),
        )
        for match in MEASURE_VALUE.finditer(text)
    ]


def extract_intervals(text: str) -> list[NumericInterval]:
    intervals: list[NumericInterval] = []
    occupied: list[tuple[int, int]] = []
    for pattern in BOUNDED_INTERVALS:
        for match in pattern.finditer(text):
            intervals.append(
                NumericInterval(
                    unit=_canonical_unit(match.group("unit")),
                    raw=match.group(0).strip(),
                    start=match.start(),
                    end=match.end(),
                    low=_number(match.group("low")),
                    high=_number(match.group("high")),
                    high_inclusive=match.group("upper") == "이하",
                )
            )
            occupied.append((match.start(), match.end()))

    def overlaps(start: int, end: int) -> bool:
        return any(start < used_end and end > used_start for used_start, used_end in occupied)

    for pattern in (OPEN_LOWER, OPEN_DASH):
        for match in pattern.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            lower = match.groupdict().get("lower")
            intervals.append(
                NumericInterval(
                    unit=_canonical_unit(match.group("unit")),
                    raw=match.group(0).strip(),
                    start=match.start(),
                    end=match.end(),
                    low=_number(match.group("low")),
                    low_inclusive=lower != "초과",
                )
            )
            occupied.append((match.start(), match.end()))
    for match in OPEN_UPPER.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        intervals.append(
            NumericInterval(
                unit=_canonical_unit(match.group("unit")),
                raw=match.group(0).strip(),
                start=match.start(),
                end=match.end(),
                high=_number(match.group("high")),
                high_inclusive=match.group("upper") == "이하",
            )
        )
    return sorted(intervals, key=lambda interval: (interval.start, interval.end))


def row_matches_query_range(query: str, row_text: str) -> bool:
    measures = extract_measures(query)
    intervals = extract_intervals(row_text)
    return any(interval.contains(measure) for measure in measures for interval in intervals)


def parse_table_range_hint(value: str) -> StructuredTableRow | None:
    match = TABLE_HINT.match(value.strip())
    if match is None:
        return None
    cells: list[StructuredCell] = []
    for raw_cell in match.group("cells").split(" | "):
        if ": " not in raw_cell:
            continue
        header, cell_value = raw_cell.split(": ", 1)
        cells.append(StructuredCell(header=header.strip(), value=cell_value.strip()))
    if not cells:
        return None
    return StructuredTableRow(
        table_id=match.group("table_id"),
        row_index=int(match.group("row_index")),
        cells=tuple(cells),
    )


def resolve_table_row(query: str, row: StructuredTableRow) -> list[TableResolution]:
    query_measures = extract_measures(query)
    matches: list[tuple[int, Measure, NumericInterval]] = []
    for cell_index, cell in enumerate(row.cells):
        for interval in extract_intervals(cell.value):
            for measure in query_measures:
                if interval.contains(measure):
                    matches.append((cell_index, measure, interval))
    resolutions: list[TableResolution] = []
    for condition_index, measure, _interval in matches:
        for output_index, output in enumerate(row.cells):
            if output_index == condition_index or not output.value:
                continue
            score = header_query_score(query, output.header)
            if score <= 0:
                continue
            resolutions.append(
                TableResolution(
                    query_measure=measure,
                    condition=row.cells[condition_index],
                    output=output,
                    target_score=score,
                    table_id=row.table_id,
                    row_index=row.row_index,
                )
            )
    return resolutions


def header_query_score(query: str, header: str) -> int:
    normalized_query = NORMALIZE_TEXT.sub("", query).casefold()
    normalized_header = NORMALIZE_TEXT.sub("", header).casefold()
    if len(normalized_header) >= 2 and normalized_header in normalized_query:
        return 1_000 + len(normalized_header)
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z가-힣]{2,}", header)
        if token not in {"현재", "구분", "내용", "질문", "답변"}
    ]
    token_score = sum(len(token) for token in tokens if token in normalized_query)
    if token_score:
        return token_score
    # Korean column names are commonly compounds (e.g. 연금소득세율).  Accept a
    # user-written suffix such as "세율" without maintaining a domain phrase list.
    overlap = _longest_common_substring(normalized_query, normalized_header)
    return len(overlap) if len(overlap) >= 2 else 0


def _longest_common_substring(left: str, right: str) -> str:
    previous = [0] * (len(right) + 1)
    best_length = 0
    best_end = 0
    for left_index, left_char in enumerate(left, start=1):
        current = [0] * (len(right) + 1)
        for right_index, right_char in enumerate(right, start=1):
            if left_char != right_char:
                continue
            current[right_index] = previous[right_index - 1] + 1
            if current[right_index] > best_length:
                best_length = current[right_index]
                best_end = left_index
        previous = current
    return left[best_end - best_length : best_end]


def matching_row_outputs(query: str, text: str) -> set[str]:
    """Return numeric outputs that share a physical row with a matching interval."""
    supported: set[str] = set()
    for line in (line for line in text.splitlines() if line.strip()):
        intervals = extract_intervals(line)
        query_measures = extract_measures(query)
        matching = [
            interval
            for interval in intervals
            if any(interval.contains(measure) for measure in query_measures)
        ]
        if not matching:
            continue
        interval_spans = [(interval.start, interval.end) for interval in intervals]
        for measure in extract_measures(line):
            if any(measure.start >= start and measure.end <= end for start, end in interval_spans):
                continue
            supported.add(normalize_measure(measure.raw))
    return supported


def normalize_measure(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", "").replace("퍼센트", "%")
