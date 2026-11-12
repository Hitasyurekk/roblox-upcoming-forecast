# roblox-up-and-coming-forecast-01 entry (MIT). Python 3.13, standard library only, no network.
#
# One forecast for the whole 72-hour window. From the history it learns how the chart moves:
# for every pair of reads 1 to 72 hours apart, where the game at each place later had been
# earlier (its rank then, or not on the chart). Averaged evenly over horizons, that gives
# P(place k later | rank r now). The last read's ranks turn it into named probabilities per
# place; whatever the history says goes to games not yet charted stays as the unnamed share.
# Deterministic: no clock, no randomness.
import json, os
from datetime import datetime

IN = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
MAXR = 100          # ranks tracked; deeper ranks and absent games share one bucket (0)
NAMES = 30          # named games per place at most
MIN_P = 0.004       # do not name a game below this
PRIOR = float(os.environ.get("FORECAST_PRIOR", "800"))  # prior strength in read pairs (backtest best 600-1500)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load():
    task = json.load(open(os.path.join(IN, "task.json"), encoding="utf-8"))
    reads = []
    for line in open(os.path.join(IN, "history.ndjson"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows = r.get("up_and_coming") or []
        if not r.get("ok", True) or len(rows) < 10:
            continue
        rows = sorted(rows, key=lambda x: x["rank"])
        reads.append((ts(r["at"]), [x["universe_id"] for x in rows]))
    reads.sort(key=lambda x: x[0])
    return task, reads


def transitions(reads, places, horizon_h=72):
    """M[k][r]: weight that place k later held the game at rank r earlier (r=0: deeper or absent)."""
    buckets = {}
    for i, (ti, ci) in enumerate(reads):
        rank = {g: n + 1 for n, g in enumerate(ci[:MAXR])}
        for tj, cj in reads[i + 1:]:
            h = (tj - ti) / 3600
            if h < 0.5 or h > horizon_h:
                continue
            b = min(int(h // 6), horizon_h // 6 - 1)
            buckets.setdefault(b, []).append([rank.get(g, 0) for g in cj[:places]])
    M = [[0.0] * (MAXR + 1) for _ in range(places)]
    npairs = sum(len(p) for p in buckets.values())
    if not buckets:
        return M, 0
    # every 6-hour horizon bucket weighs the same, so the mix matches a uniform 72-hour window
    for pairs in buckets.values():
        w = 1.0 / len(pairs) / len(buckets)
        for row in pairs:
            for k, r in enumerate(row):
                M[k][r] += w
    return M, npairs


def smooth(M, places, npairs=0, strength=PRIOR):
    """Blend each place's row with a rank kernel prior so unseen (k, r) cells keep some mass.
    The rows are normalised over horizons, so the data's weight comes from how many read
    pairs stand behind them: few pairs lean on the prior, many on the history."""
    out = []
    for k in range(places):
        row = M[k]
        seen = sum(row)
        prior = [0.0] * (MAXR + 1)
        for r in range(1, MAXR + 1):
            prior[r] = 1.0 / (1 + abs(r - (k + 1)) * 0.6) ** 2
        prior[0] = 0.35 * sum(prior[1:])
        ps = sum(prior)
        prior = [x / ps for x in prior]
        a = strength / (strength + npairs)  # little data: lean on the prior
        tot = seen or 1.0
        out.append([(1 - a) * (row[r] / tot) + a * prior[r] for r in range(MAXR + 1)])
    return out


def main():
    task, reads = load()
    places = task.get("places", 10)
    last = reads[-1][1]
    M, npairs = transitions(reads, places)
    P = smooth(M, places, npairs)
    probs = []
    for k in range(places):
        cand = sorted(((P[k][r], r) for r in range(1, min(MAXR, len(last)) + 1)), reverse=True)
        named = [(p, r) for p, r in cand[:NAMES] if p >= MIN_P]
        s = sum(p for p, _ in named)
        if s > 0.97:  # always leave a little for games not named here
            named = [(p * 0.97 / s, r) for p, r in named]
        for p, r in named:
            probs.append({"universe_id": last[r - 1], "place": k + 1, "p": round(p, 6)})
    # best guess at the order: each place's most likely game, without repeats
    top, used = [], set()
    for k in range(places):
        for p, r in sorted(((P[k][r], r) for r in range(1, len(last) + 1)), reverse=True):
            g = last[r - 1]
            if g not in used:
                top.append(g); used.add(g); break
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        json.dump({"probabilities": probs, "top10": top}, f)


if __name__ == "__main__":
    main()
