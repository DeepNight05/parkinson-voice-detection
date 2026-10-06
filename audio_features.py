"""Extract voice features from a recording with Praat (via parselmouth).

Produces the 16 features that correspond to the UCI Parkinson's columns
which Praat can measure. The remaining 6 UCI features (RPDE, DFA, spread1,
spread2, D2, PPE) are nonlinear measures and are not computed here.

Caveat: the UCI data was produced with Kay Pentax MDVP software, so Praat
values are close in meaning but not identical. Treat results as a demo.
"""
import os
import tempfile

import numpy as np
import parselmouth
from parselmouth.praat import call

F0_MIN, F0_MAX = 75, 500


def extract_features(audio_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        snd = parselmouth.Sound(path)
    finally:
        os.remove(path)

    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    if snd.duration < 1.0:
        raise ValueError("Recording is too short. Please record at least 2-3 seconds.")

    pitch = call(snd, "To Pitch", 0.0, F0_MIN, F0_MAX)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 20:
        raise ValueError("Could not find a steady voiced signal. "
                         "Hold a clear 'aaah' sound and reduce background noise.")

    pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)

    def jit(kind):
        return call(pp, kind, 0, 0, 0.0001, 0.02, 1.3)

    def shim(kind):
        return call([snd, pp], kind, 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    feats = {
        "MDVP:Fo(Hz)": float(np.mean(f0)),
        "MDVP:Fhi(Hz)": float(np.max(f0)),
        "MDVP:Flo(Hz)": float(np.min(f0)),
        "MDVP:Jitter(%)": jit("Get jitter (local)"),
        "MDVP:Jitter(Abs)": jit("Get jitter (local, absolute)"),
        "MDVP:RAP": jit("Get jitter (rap)"),
        "MDVP:PPQ": jit("Get jitter (ppq5)"),
        "Jitter:DDP": jit("Get jitter (ddp)"),
        "MDVP:Shimmer": shim("Get shimmer (local)"),
        "MDVP:Shimmer(dB)": shim("Get shimmer (local_dB)"),
        "Shimmer:APQ3": shim("Get shimmer (apq3)"),
        "Shimmer:APQ5": shim("Get shimmer (apq5)"),
        "MDVP:APQ": shim("Get shimmer (apq11)"),
        "Shimmer:DDA": shim("Get shimmer (dda)"),
        "HNR": float(hnr),
        # NHR is approximated from HNR (noise-to-harmonics power ratio).
        "NHR": float(10 ** (-hnr / 10)),
    }
    bad = [k for k, v in feats.items() if v is None or not np.isfinite(v)]
    if bad:
        raise ValueError("Praat could not measure: " + ", ".join(bad) +
                         ". Try a longer, cleaner recording.")
    return feats
