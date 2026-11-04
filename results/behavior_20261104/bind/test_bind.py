"""Counterexample tests for Phi_bind v1 (Design_New round 3 §5 + Design_PH §1/§3). python test_bind.py"""
from bind_scorer import score

W, H = 2000, 1000
VAL, ROWH, COLH = [0.60, 0.500, 0.66, 0.505], [0.05, 0.500, 0.20, 0.505], [0.60, 0.20, 0.66, 0.205]


def slot(value="1,263,186", path=(("Year 4",),), alone=(0,), other_cols=("year 1", "year 2", "year 3"), boxes=(VAL, ROWH, COLH)):
    return {"value": value, "row": ["Unlevered Cash Flow"], "path": [list(p) for p in path], "alone": list(alone),
            "other_rows": ["Net Income"], "other_cols": list(other_cols), "boxes": [list(b) for b in boxes]}


T = {"slots": [slot()]}
ALL = [(0, 0, 2000, 600)]                       # wide crop: headers + row label legible? (checked below)
CROP = [(1150, 470, 1350, 530), (50, 470, 450, 530), (1150, 180, 1350, 230)]   # value, row header, col header
JUNK = [(0, 700, 300, 900), (1500, 800, 1800, 1000)]
ONLYVAL = [(1150, 470, 1350, 530)]
WRAP = "<think>\nStep {s} | commit\n{body}\n</think>\n<answer>B</answer>"
GOOD = WRAP.format(s=3, body="The Unlevered Cash Flow row under Year 4 reads 1,263,186 [observation_1].")


def E(entry, crops, text):
    return score(entry, (W, H), crops, text)[0]


checks = [
    ("correct binding, value + both headers legible", E(T, CROP, GOOD) == 1.0),
    ("extra irrelevant / duplicate crops add nothing", E(T, CROP + JUNK + CROP, GOOD) == E(T, CROP, GOOD)),
    ("headers never legible -> 0 (value alone not enough)", E(T, ONLYVAL, GOOD) == 0.0),
    ("protocol metadata only (Step 3 / observation_2) -> 0",
     E({"slots": [slot(value="3")]}, CROP, WRAP.format(s=3, body="The scene is unclear [observation_2].")) == 0.0),
    ("Step 4 + 'Year 4' is not a reading of value 4",
     E({"slots": [slot(value="4")]}, CROP, WRAP.format(s=4, body="The Unlevered Cash Flow row under Year 4 is blank.")) == 0.0),
    ("swapped column -> 0",
     E(T, CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row under Year 3 reads 1,263,186.")) == 0.0),
    ("value bound to another row -> 0",
     E(T, CROP, WRAP.format(s=3, body="Under Year 4, Net Income reads 1,263,186.")) == 0.0),
    ("same turn: right first, later adopts a wrong value -> 0",
     E(T, CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row under Year 4 reads 1,263,186. "
                                       "Re-checking, the Unlevered Cash Flow under Year 4 is 1,509,598.")) == 0.0),
    ("same turn: wrong first, later corrected -> 1",
     E(T, CROP, WRAP.format(s=3, body="The Unlevered Cash Flow under Year 4 looked like 1,509,598. "
                                       "Correcting: the Unlevered Cash Flow row under Year 4 reads 1,263,186.")) == 1.0),
    ("enumeration nearest to column is another value -> 0",
     E(T, CROP, WRAP.format(s=3, body="Unlevered Cash Flow: Year 4 1,509,598, then 1,263,186 and 1,925,852.")) == 0.0),
    ("ambiguous multi-level path: month alone (other year has it) -> 0",
     E({"slots": [slot(path=(("2025",), ("Nov", "Month 11")), alone=(0,), other_cols=("2026", "nov", "month 11"))]},
       CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row under Nov reads 1,263,186.")) == 0.0),
    ("ambiguous multi-level path: year + month -> 1",
     E({"slots": [slot(path=(("2025",), ("Nov", "Month 11")), alone=(0,), other_cols=("2026", "nov", "month 11"))]},
       CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row under 2025 Nov reads 1,263,186.")) == 1.0),
    ("synonym levels: one unambiguous level suffices -> 1",
     E({"slots": [slot(path=(("Aug",), ("Month 8",)), alone=(0, 1), other_cols=("sep", "month 9"))]},
       CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row under Aug reads 1,263,186.")) == 1.0),
    ("final answer turn only (earlier turn right, final wrong) -> 0",
     E(T, CROP, "<think>\nStep 1 | locate\nUnlevered Cash Flow under Year 4 reads 1,263,186.\n</think>\n"
                "<grounding>{\"bbox_2d\":[0.5,0.4,0.7,0.6],\"source\":\"original_image\"}</grounding>"
                + WRAP.format(s=2, body="The Unlevered Cash Flow under Year 4 reads 1,925,852.")) == 0.0),
    ("v2: whitespace label variant FY2024 vs FY 2024 -> 1",
     E({"slots": [slot(path=(("FY 2024",),), other_cols=("fy 2023",))]}, CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row under FY2024 is 1,263,186.")) == 1.0),
    ("v2: one column, several rows (value next to its row) -> 1",
     E(T, CROP, WRAP.format(s=3, body="The Year 4 column shows Net Income at 1,925,852 and Unlevered Cash Flow at 1,263,186.")) == 1.0),
    ("v2: one column, several rows, target row given another's value -> 0",
     E(T, CROP, WRAP.format(s=3, body="The Year 4 column shows Unlevered Cash Flow at 1,925,852 and Net Income at 1,263,186.")) == 0.0),
    ("v2: explicit later adoption in a listed row -> adopted value",
     E(T, CROP, WRAP.format(s=3, body="The Unlevered Cash Flow row is 1,509,598, 1,263,186, with Year 4 as the third column, so its value is 1,925,852.")) == 0.0),
    ("v2: unicode minus and leading-dot decimals",
     E({"slots": [slot(value=".85")]}, CROP, WRAP.format(s=3, body="Unlevered Cash Flow under Year 4 = 0.85.")) == 1.0),
    ("accounting parentheses equal value -> 1",
     E({"slots": [slot(value="(144,489)")]}, CROP, WRAP.format(s=3, body="The Unlevered Cash Flow under Year 4 is (144,489).")) == 1.0),
]
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
assert all(ok for _, ok in checks)
print(f"bind v1 tests: {len(checks)}/{len(checks)} pass")
