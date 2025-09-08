import yaml, sys, io, os
src = sys.argv[1]  # path to inference.yml
dst = sys.argv[2]  # path to write ppocr_keys_ocrv5_custom.txt
with io.open(src, "r", encoding="utf-8") as f:
    y = yaml.safe_load(f)
chars = y["PostProcess"]["character_dict"]
#os.makedirs(os.path.dirname(dst), exist_ok=True)
with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
    for ch in chars:
        f.write(str(ch) + "\n")
print(f"wrote {len(chars)} characters to {dst}")
