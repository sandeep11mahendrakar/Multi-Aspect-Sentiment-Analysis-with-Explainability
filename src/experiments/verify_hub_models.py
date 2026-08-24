"""Verify the Hub-uploaded Lilly models load and behave identically (pull-as-a-user test)."""
import os

assert os.environ.get("DOC_MODEL_ID"), "set DOC_MODEL_ID"
assert os.environ.get("ABSA_MODEL_ID"), "set ABSA_MODEL_ID"

import numpy as np

from apps.webapp.backend import inference
from apps.webapp.backend.scoring import classify

m = inference.load_models()
print("hub models loaded OK:", os.environ["DOC_MODEL_ID"], "+", os.environ["ABSA_MODEL_ID"])

trio = [
    ("Absolutely love this dress! The fabric is soft, the print is gorgeous, and the fit is perfect.",
     "expect pos committed", False),
    ("Very disappointed. Cheap material, loose threads everywhere, returned it.",
     "expect neg committed", False),
    ("beautiful pattern but runs extremely small", "expect UNSURE", True),
]
ok = True
for text, expect, want_unsure in trio:
    p = inference.doc_probs(m, [text])[0]
    r = classify(p)
    a = inference.absa_analyze(m, text)
    match = r["unsure"] == want_unsure
    ok &= match
    print(f"{expect}: probs={np.round(p, 4).tolist()} -> score={r['score']} "
          f"band={r['band']} unsure={r['unsure']} conf={r['confidence']:.4f} "
          f"aspects={a} match={match}")
print("HUB VERIFICATION:", "PASS" if ok else "FAIL")
