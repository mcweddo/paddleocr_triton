import io
import os
import yaml
import numpy as np


def load_charset_and_blank_from_yaml(yaml_path: str):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"inference.yml not found: {yaml_path}")
    with io.open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    pp = y.get("PostProcess", {}) or {}

    chars = pp.get("character_dict")
    if isinstance(chars, list) and chars:
        charset = [str(c).strip("\n").strip("\r\n") for c in chars]
    else:
        dict_path = pp.get("character_dict_path")
        if not dict_path or not os.path.isfile(dict_path):
            raise ValueError("No character_dict in YAML and character_dict_path missing.")
        with io.open(dict_path, "r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]

    blank_at_zero = False
    if "blank_at_zero" in pp:
        blank_at_zero = bool(pp["blank_at_zero"])
    elif "ctc_blank" in pp:
        blank_at_zero = int(pp["ctc_blank"]) == 0
    elif "blank_index" in pp:
        blank_at_zero = int(pp["blank_index"]) == 0

    charset = ['blank'] + charset

    return charset, blank_at_zero


class CTCLabelDecodeRobust:
    def __init__(self, character_list, blank_at_zero, merge_repeats=True):
        # character_list: list[str] without the CTC blank
        self.character = character_list[:]  # length = K
        self.merge_repeats = merge_repeats
        self.blank_at_zero = bool(blank_at_zero)

    def _normalize_logits_shape(self, logits, vocab_size=None):

        arr = np.asarray(logits)
        # Squeeze a leading batch of 1 if present
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for 3D logits, got shape {arr.shape}")
            arr = arr[0]

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D logits after squeeze, got shape {arr.shape}")

        h, w = arr.shape
        if isinstance(vocab_size, (int, np.integer)) and vocab_size >= 2:
            if w == vocab_size and h != vocab_size:
                # already [T, V]
                print("It is [T, V]")
                pass
            elif h == vocab_size and w != vocab_size:
                # it's [V, T] -> transpose
                arr = arr.T
                h, w = arr.shape
                print("It is [V,T]")
            elif h == vocab_size and w == vocab_size:
                # square & ambiguous; leave as-is (no transpose)
                pass
            else:
                # Neither axis matches vocab_size → fall back to heuristic below.
                if h > w:
                    # larger dim likely vocab -> transpose to make second dim V
                    arr = arr.T
                    h, w = arr.shape
                    print("It is [V,T]")
        else:
            # 2) Heuristic: vocab (V) is typically the larger axis; time (T) the smaller.
            if h > w:
                arr = arr.T
                h, w = arr.shape
                print("It is [V,T]")

        T, V = arr.shape
        if V < 2:
            raise ValueError(f"Logits look wrong after normalization: [T,V]=[{T},{V}]")
        return arr.astype(np.float32, copy=False)

    def decode(self, logits: np.ndarray):
        V_expected = len(self.character) + 1  # blank at V-1
        probs = self._normalize_logits_shape(logits, vocab_size=V_expected)  # [T,V]
        T, V = probs.shape
        blank_idx = 0 if self.blank_at_zero else (V - 1)
        K = len(self.character)

        if K != blank_idx:
            # Not raising; it’s common when folks change models/dicts
            # You can print/log here if you want visibility
            pass

        prev = -1
        text_chars = []
        confs = []

        idxs = probs.argmax(axis=1)  # [T]
        maxp = probs.max(axis=1)     # [T]

        for t, cls in enumerate(idxs.tolist()):
            if cls == blank_idx:
                prev = cls
                continue
            if self.merge_repeats and cls == prev:
                prev = cls
                continue
            if 0 <= cls < K:
                text_chars.append(self.character[cls])
                confs.append(float(maxp[t]))
            else:
                prev = cls
                continue
            prev = cls

        if not text_chars:
            return "", np.nan
        if not confs:
            score = np.nan
        else:
            score = float(np.median(np.asarray(confs, dtype=np.float32)))
        return "".join(text_chars), score

