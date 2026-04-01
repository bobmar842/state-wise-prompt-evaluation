import json, glob

run1 = {}
for f in glob.glob("output/run1_reports/report_*.json"):
    r = json.load(open(f, encoding="utf-8"))
    run1[r["persona_id"]] = r["grand_overall"]

run2 = {}
for f in glob.glob("output/report_*.json"):
    r = json.load(open(f, encoding="utf-8"))
    if f not in glob.glob("output/run1_reports/*"):
        run2[r["persona_id"]] = r["grand_overall"]

print(f"{'Persona':<25} {'Run1':>6} {'Run2':>6} {'Diff':>6}")
print("-" * 45)
for pid in sorted(run1.keys()):
    r1 = run1.get(pid, 0)
    r2 = run2.get(pid, 0)
    diff = abs(r1 - r2)
    flag = " ⚠️" if diff > 0.1 else ""
    print(f"{pid:<25} {r1:>6.3f} {r2:>6.3f} {diff:>6.3f}{flag}")