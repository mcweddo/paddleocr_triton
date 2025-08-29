import io
import os
import yaml
import numpy as np


def load_charset_from_yaml(yaml_path: str):

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"inference.yml not found: {yaml_path}")

    with io.open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    pp = (y or {}).get("PostProcess", {})
    chars = pp.get("character_dict")
    if isinstance(chars, list) and len(chars) > 0:
        return [str(c).strip("\n").strip("\r\n") for c in chars]

    dict_path = pp.get("character_dict_path")
    if dict_path and os.path.isfile(dict_path):
        with io.open(dict_path, "r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]

    raise ValueError(
        "No character_dict found in YAML and character_dict_path missing/not readable."
    )

class CTCLabelDecodeRobust:
    def __init__(self, character_list, merge_repeats=True, use_space_char=True):
        # character_list: list[str] without the CTC blank
        self.character = character_list[:]  # length = K
        self.merge_repeats = merge_repeats
        self.use_space_char = use_space_char

    def _normalize_logits_shape(self, logits, vocab_size=None):
        """
        Normalize logits to shape [T, V] (float32).

        Accepts:
          - [T, V]
          - [V, T]
          - [B, T, V] with B==1   (squeezed to [T, V])

        Args:
          logits: np.ndarray-like
          vocab_size: Optional[int] = len(charset) + 1 (CTC blank)

        Returns:
          np.ndarray of shape [T, V], dtype float32
        """

        arr = np.asarray(logits)
        # Squeeze a leading batch of 1 if present
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for 3D logits, got shape {arr.shape}")
            arr = arr[0]

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D logits after squeeze, got shape {arr.shape}")

        h, w = arr.shape  # unknown which is T or V yet

        # 1) If we know vocab_size, prefer that to decide orientation.
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

        # Final sanity: now interpret as [T, V]
        T, V = arr.shape
        if V < 2:
            raise ValueError(f"Logits look wrong after normalization: [T,V]=[{T},{V}]")
        return arr.astype(np.float32, copy=False)

    def decode(self, logits: np.ndarray):
        V_expected = len(self.character) + 1  # blank at V-1
        probs = self._normalize_logits_shape(logits, vocab_size=V_expected)  # [T,V]
        T, V = probs.shape
        blank_idx = V - 1
        K = len(self.character)

        if K != blank_idx:
            # Not raising; it’s common when folks change models/dicts
            # You can print/log here if you want visibility
            pass

        # Greedy decode
        prev = -1
        text_chars = []
        confs = []

        # argmax along classes
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

# === factory/helper ===
def load_charset(dict_path: str):
    # Typical Paddle dicts are one char per line; ensure UTF-8
    with open(dict_path, "r", encoding="utf-8") as f:
        chars = [line.rstrip("\n\r") for line in f]
    # Optionally add space char if your training used it but the file doesn’t contain it
    return chars
