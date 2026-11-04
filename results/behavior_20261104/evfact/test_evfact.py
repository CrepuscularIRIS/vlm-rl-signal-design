"""Gate 1: counterexample requirements for the evidence-fact scorer (Design_New round 3 §5) + guards.
python test_evfact.py"""
from evfact import score

W, H = 2000, 1000
TABLE = {"slots": [{"type": "table", "value": "1,263,186", "kind": "number", "box": [0.60, 0.500, 0.66, 0.505],
                    "row": "Unlevered Cash Flow", "cols": ["Year 4"], "other_rows": ["Net Income"],
                    "other_cols": ["Year 1", "Year 2", "Year 3"]}]}
ENT = {"slots": [{"type": "entity", "value": "BROWN", "kind": "text", "box": [0.10, 0.10, 0.20, 0.13]}]}
BIGTABLE = {"slots": [dict(TABLE["slots"][0], box=[0.30, 0.40, 0.70, 0.46])]}  # legible at the original view

CROP = [(1150, 470, 1350, 530)]                       # covers the value box, magnified
JUNK = [(0, 0, 300, 200), (1500, 800, 1800, 1000)]     # irrelevant crops
BIG1 = [(1100, 450, 1400, 560)]                        # one crop with the same evidence
GOOD = "<think>\nStep 3 | commit\nThe Unlevered Cash Flow row under Year 4 reads 1,263,186 [observation_1].\n</think>\n<answer>B</answer>"
SWAPCOL = ("<think>\nStep 3 | commit\nThe Unlevered Cash Flow row under Year 3 reads 1,263,186, while Year 4 is shown "
           "at the far right of the long header line [observation_1].\n</think>\n<answer>C</answer>")
WRONGROW = "<think>\nStep 3 | commit\nThe Net Income row under Year 4 reads 1,263,186 [observation_1].\n</think>\n<answer>C</answer>"
ENUM = "<think>\nStep 3 | commit\nThe table shows 1,263,186, 1,509,598, 1,925,852 [observation_1].\n</think>\n<answer>C</answer>"
NEG = "<think>\nStep 3 | commit\nThe Unlevered Cash Flow row under Year 4 is not 1,263,186 [observation_1].\n</think>\n<answer>C</answer>"
EARLY_RIGHT_FINAL_WRONG = ("<think>\nStep 1 | locate\nUnlevered Cash Flow under Year 4 reads 1,263,186.\n</think>\n"
                           "<grounding>{\"bbox_2d\":[0.5,0.4,0.7,0.6],\"source\":\"original_image\"}</grounding>"
                           "<think>\nStep 2 | commit\nThe Net Income row under Year 4 reads 1,263,186.\n</think>\n<answer>C</answer>")
E_GOOD = "<think>\nStep 2 | commit\nThe name tag reads BROWN [observation_1].\n</think>\n<answer>A</answer>"
E_SUB = "<think>\nStep 2 | commit\nThe name tag reads BROWNE [observation_1].\n</think>\n<answer>A</answer>"
E_NEG = "<think>\nStep 2 | commit\nThe name tag is not BROWN [observation_1].\n</think>\n<answer>A</answer>"
ECROP = [(150, 80, 450, 160)]


def E(entry, crops, text):
    return score(entry, (W, H), crops, text)[0]


checks = [
    ("correct fact, supported", E(TABLE, CROP, GOOD) == 1.0),
    ("extra irrelevant / duplicate crops do not add", E(TABLE, CROP + JUNK + CROP, GOOD) == E(TABLE, CROP, GOOD)),
    ("one crop vs three crops, same evidence -> same", E(TABLE, BIG1, GOOD) == E(TABLE, JUNK + CROP, GOOD)),
    ("same crops, swapped column binding -> lower", E(TABLE, CROP, SWAPCOL) < E(TABLE, CROP, GOOD)),
    ("same crops, value bound to wrong row -> 0", E(TABLE, CROP, WRONGROW) == 0.0),
    ("fixing binding without extra crops -> higher", E(TABLE, CROP, GOOD) > E(TABLE, CROP, WRONGROW)),
    ("never seen at usable scale -> 0", E(TABLE, [], GOOD) == 0.0),
    ("original view already sufficient: ritual crop adds nothing", E(BIGTABLE, [], GOOD) == E(BIGTABLE, CROP, GOOD) == 1.0),
    ("early correct, final wrong -> 0 (final answer turn only)", E(TABLE, CROP, EARLY_RIGHT_FINAL_WRONG) == 0.0),
    ("unlabelled enumeration -> 0", E(TABLE, CROP, ENUM) == 0.0),
    ("negated value -> 0", E(TABLE, CROP, NEG) == 0.0),
    ("entity: exact value supported -> 1", E(ENT, ECROP, E_GOOD) == 1.0),
    ("entity: superstring BROWNE -> 0", E(ENT, ECROP, E_SUB) == 0.0),
    ("entity: negated -> 0", E(ENT, ECROP, E_NEG) == 0.0),
]
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
assert all(ok for _, ok in checks), "gate 1 failed"
print(f"gate 1: {len(checks)}/{len(checks)} pass; span:", score(TABLE, (W, H), CROP, GOOD)[2])
