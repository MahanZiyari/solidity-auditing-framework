import json
from pathlib import Path
from phase1_runner import run_phase1
from slither_metadata_extractor import build_planner_prompt

manifest = run_phase1("contracts/SimpleVault.sol", output_dir="phase1_output")

messages = []
for row in manifest["contracts"]:
    if row["slither"] is None or row["slither"]["status"] != "success":
        continue                                  # این قرارداد برای Planner قابلاستفاده نیست
    metadata = json.loads(Path(row["slither"]["output"]).read_text(encoding="utf-8"))
    ast      = json.loads(Path(row["antlr"]["output"]).read_text(encoding="utf-8"))
    prompt   = build_planner_prompt(metadata, ast)   # -> {"system": str, "user": str}
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user",   "content": prompt["user"]},
    ]

for message in messages:
    print(message["content"])
