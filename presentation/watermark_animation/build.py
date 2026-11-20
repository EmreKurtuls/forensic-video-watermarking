import json
from pathlib import Path

HERE = Path(__file__).parent
d = json.loads((HERE / "data.json").read_text())
c = d["close"]
a = d["algs"]["PRESENT-128"]
values = {
    **{k: v for k, v in d.items() if isinstance(v, (str, int, float))},
    **{k: v for k, v in a.items() if isinstance(v, (str, int, float))},
    "plaintext_hex": f"{int(d['block_bits'], 2):016x}",
    "nblocks_fmt": f"{d['nblocks']:,}",
    "per_bit": round(d["nblocks"] / 188),
    "share_pct": round(a["share_changed"] * 100),
    "grid_rows": d["blocks_y"] * 8,
    "edge_rows": d["height"] - d["blocks_y"] * 8,
    "close_row": c["row"], "close_col": c["col"], "close_bit_index": c["bit_index"], "close_bit": a["close_bit"],
    "close_alpha": c["alpha_b"], "close_s": "+1" if a["close_bit"] else "−1",
    "close_u": "[" + ", ".join("+1" if x > 0 else "−1" for x in c["u"]) + "]",
}
html = (HERE / "template.html").read_text()
for k, v in values.items():
    html = html.replace("{{" + k + "}}", str(v))
assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
html = html.replace("__DATA__", json.dumps(d, separators=(",", ":")))
(HERE / "index.html").write_text(html)
print("wrote", HERE / "index.html")
