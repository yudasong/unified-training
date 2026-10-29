"""Deterministic verbalized solver for nl_tower (the oracle that writes the midtraining traces).

Candidate-set elimination plus REFERENCE RESOLUTION: a clue whose term is
"the best friend of the person on floor 19" can only be used once floor 19's occupant is fixed; the
trace then narrates the resolution ("Floor 19 is Zoe's, so the best friend of the person on floor 19 is
Liam.") before applying the rule.  Terms in numeric positions may resolve to a floor constant without
an identity ("Ava = 12, so the person directly above Ava is on floor 19.").

solve_verbal(names, domain, fr, clues, style=0, order_seed=0, unary="mixed") ->
  {"solution": [floors] | None, "lines": [...], "fingerprint": {...}}
No branching: returns None when propagation stalls (the generator rejects such puzzles).

This solver is both the generator's acceptance filter and the writer of every midtraining trace (no language model
writes any trace). The generator helpers are imported inside solve_verbal() to avoid a circular import."""

import random


def _fmt(s):
    return "{" + ", ".join(map(str, sorted(s))) + "}"


STY = [
    {
        "par": "{n} lives on an {w} floor, so {n} can only be {s}.",
        "ext": "{n} lives on the {w} floor, so {n} = {v}.",
        "few": "From {d}, the only options are ({a},{b}) in {ps}.",
        "many": "Applying {d} narrows things down: {a} in {sa}, {b} in {sb}.",
        "una": "From {d}, {a} can only be {s}.",
        "fix": "So {n} = {v}. No one else can have {v}.",
        "res_friend": "{x}'s best friend is {y}.",
        "res_occ": "Floor {f} is {x}'s, so {t} is {x}.",
        "res_next": "{x} = {v}, so the floor directly {dir} {x} is {f}.",
        "res_rank": "The {w} floor is {f}.",
        "res_holder": "Floor {f} is {x}'s.",
        "res_sum": "So {t} is {y}.",
    },
    {
        "par": "Since {n} must be on an {w} floor, {n} is one of {s}.",
        "ext": "Because {n} is on the {w} floor, {n} = {v}.",
        "few": "The clue {d} leaves these pairs for ({a},{b}): {ps}.",
        "many": "Using {d}: {a} is now in {sa} and {b} in {sb}.",
        "una": "Using {d}: {a} is now in {s}.",
        "fix": "That pins {n} to {v}, so {v} is taken.",
        "res_friend": "{x}'s best friend is {y}.",
        "res_occ": "{x} is on floor {f}, so {t} means {x}.",
        "res_next": "{x} is on {v}, so the floor directly {dir} {x} is {f}.",
        "res_rank": "The {w} floor is {f}.",
        "res_holder": "{x} is on floor {f}.",
        "res_sum": "Hence {t} is {y}.",
    },
]


def _pairs(kind, ci, cj, arg):
    if kind == "mult":
        return [(a, b) for a in ci for b in cj if a == arg * b]
    if kind == "sum":
        return [(a, b) for a in ci for b in cj if a + b == arg]
    if kind == "diff":
        return [(a, b) for a in ci for b in cj if a - b == arg]
    if kind == "gt":
        return [(a, b) for a in ci for b in cj if a > b]
    raise ValueError(kind)


def solve_verbal(names, domain, fr, clues, style=0, order_seed=0, unary="mixed"):
    from .generator import DEFERRED, clue_pattern, next_floor, rank_word, term_text

    sty = STY[style]
    n = len(names)
    domain = list(domain)
    cand = [set(domain) for _ in range(n)]
    fixed = [None] * n
    holder = {}
    lines = []
    resolved_txt = set()  # (term repr) already narrated
    first_applied = {}  # clue index -> (round, n_fixed_before)
    fixes_in_order = []

    def fix(i, v):
        fixed[i] = v
        cand[i] = {v}
        holder[v] = i
        fixes_in_order.append(i)
        lines.append(sty["fix"].format(n=names[i], v=v))
        for k in range(n):
            if k != i:
                cand[k].discard(v)

    def res_name(t, narr):
        """-> name idx or None. narr: list collecting resolution lines (emitted only if the clue applies)."""
        k = t[0]
        if k == "n":
            return t[1]
        if k == "friend":
            x = res_name(t[1], narr)
            if x is None:
                return None
            y = fr[x]
            key = ("friend", repr(t))
            if key not in resolved_txt:
                if t[1][0] == "n":
                    narr.append((key, sty["res_friend"].format(x=names[x], y=names[y])))
                else:
                    narr.append((key, sty["res_sum"].format(t=term_text(t, names, domain), y=names[y])))
            return y
        if k in ("occ", "rank"):
            f = t[1] if k == "occ" else domain[t[1]]
            x = holder.get(f)
            if x is None:
                return None
            key = ("holder", repr(t))
            if key not in resolved_txt:
                if k == "rank":
                    narr.append((key + ("r",), sty["res_rank"].format(w=rank_word(domain, t[1]), f=f)))
                narr.append((key, sty["res_occ"].format(f=f, x=names[x], t=term_text(t, names, domain))))
            return x
        if k in ("above", "below"):
            f = res_floor(t, narr)
            if f is None:
                return None
            x = holder.get(f)
            if x is None:
                return None
            key = ("holder", repr(t))
            if key not in resolved_txt:
                narr.append((key, sty["res_occ"].format(f=f, x=names[x], t=term_text(t, names, domain))))
            return x
        raise ValueError(k)

    def res_floor(t, narr):
        """-> ("var", i) | ("const", v) | None ; for above/below returns the const floor (int) when known."""
        k = t[0]
        if k == "occ":
            return t[1]
        if k == "rank":
            key = ("rank", repr(t))
            if key not in resolved_txt:
                narr.append((key, sty["res_rank"].format(w=rank_word(domain, t[1]), f=domain[t[1]])))
            return domain[t[1]]
        if k in ("above", "below"):
            inner = t[1]
            if inner[0] in ("occ", "rank"):
                v = res_floor(inner, narr)
            else:
                x = res_name(inner, narr)
                if x is None or fixed[x] is None:
                    return None
                v = fixed[x]
            if v is None:
                return None
            f = next_floor(domain, v, up=(k == "above"))
            if f is None:
                return None
            key = ("next", repr(t))
            if key not in resolved_txt:
                inner_txt = term_text(inner, names, domain)
                narr.append(
                    (key, sty["res_next"].format(x=inner_txt, v=v, dir="above" if k == "above" else "below", f=f))
                )
            return f
        x = res_name(t, narr)
        return None if x is None else ("var", x)

    def as_pos(t, narr):
        r = res_floor(t, narr)
        if r is None:
            return None
        if isinstance(r, tuple):
            return r
        return ("const", r)

    def apply_con(ci, c, rnd):
        kind = c[0]
        narr = []
        changed = False
        if kind in ("parity", "extreme", "rankc"):
            p = as_pos(c[1], narr)
            if p is None or p[0] == "const":
                return False
            i = p[1]
            if kind == "rankc":
                v = domain[c[3]]
                new = {v} & cand[i]
                if new != cand[i]:
                    _emit(narr)
                    lines.append(sty["ext"].format(n=names[i], w=rank_word(domain, c[3]), v=v))
                    cand[i] = new
                    changed = True
            elif kind == "parity":
                new = {v for v in cand[i] if v % 2 == c[3]}
                if new != cand[i]:
                    _emit(narr)
                    lines.append(sty["par"].format(n=names[i], w="odd" if c[3] else "even", s=_fmt(new)))
                    cand[i] = new
                    changed = True
            else:
                v = domain[-1] if c[3] == "high" else domain[0]
                new = {v} & cand[i]
                if new != cand[i]:
                    _emit(narr)
                    lines.append(sty["ext"].format(n=names[i], w="highest" if c[3] == "high" else "lowest", v=v))
                    cand[i] = new
                    changed = True
        else:
            p = as_pos(c[1], narr)
            q = as_pos(c[2], narr)
            if p is None or q is None:
                return False
            arg = c[3]
            ta = term_text(c[1], names, domain)
            tb = term_text(c[2], names, domain)
            sym = {
                "mult": f"{ta} = {arg} x {tb}",
                "sum": f"{ta} + {tb} = {arg}",
                "diff": f"{ta} - {tb} = {arg}",
                "gt": f"{ta} > {tb}",
            }[kind]
            if p[0] == "var" and q[0] == "var":
                i, j = p[1], q[1]
                if i == j:
                    return False
                ps = _pairs(kind, cand[i], cand[j], arg)
                ps = [(a, b) for a, b in ps if a != b]
                ni = {a for a, b in ps}
                nj = {b for a, b in ps}
                if ni != cand[i] or nj != cand[j]:
                    _emit(narr)
                    if 0 < len(ps) <= 5:
                        lines.append(
                            sty["few"].format(
                                d=sym,
                                a=names[i],
                                b=names[j],
                                ps="{" + ", ".join(f"({a},{b})" for a, b in sorted(ps)) + "}",
                            )
                        )
                    else:
                        lines.append(sty["many"].format(d=sym, a=names[i], b=names[j], sa=_fmt(ni), sb=_fmt(nj)))
                    cand[i] = ni
                    cand[j] = nj
                    changed = True
            elif p[0] == "const" and q[0] == "const":
                return False
            else:
                # one side is a known floor constant -> unary filter on the other
                if p[0] == "var":
                    i, v = p[1], q[1]
                    new = {a for a in cand[i] if _pairs(kind, {a}, {v}, arg)}
                else:
                    i, v = q[1], p[1]
                    new = {b for b in cand[i] if _pairs(kind, {v}, {b}, arg)}
                new.discard(v)
                if new != cand[i]:
                    _emit(narr)
                    sym2 = sym.replace(ta, f"floor {p[1]}" if p[0] == "const" else ta).replace(
                        tb, f"floor {q[1]}" if q[0] == "const" else tb
                    )
                    lines.append(sty["una"].format(d=sym2, a=names[i], s=_fmt(new)))
                    cand[i] = new
                    changed = True
        if changed and ci not in first_applied:
            first_applied[ci] = (rnd, len(fixes_in_order))
        return changed

    def _emit(narr):
        for key, txt in narr:
            if key not in resolved_txt:
                resolved_txt.add(key)
                lines.append(txt)

    order = list(range(len(clues)))
    if order_seed:
        random.Random(order_seed).shuffle(order)
    if unary == "unary_first":
        order.sort(key=lambda i: clues[i][0] not in ("parity", "extreme", "rankc"))
    elif unary == "binary_first":
        order.sort(key=lambda i: clues[i][0] in ("parity", "extreme", "rankc"))

    rounds = 0
    for rnd in range(60):
        rounds = rnd + 1
        changed = False
        for ci in order:
            if apply_con(ci, clues[ci], rnd):
                changed = True
            for i in range(n):
                if not cand[i]:
                    return {"solution": None, "lines": lines, "fingerprint": {"fail": "empty"}}
                if fixed[i] is None and len(cand[i]) == 1:
                    fix(i, next(iter(cand[i])))
                    changed = True
        # a floor left to exactly one person
        for v in domain:
            if v in holder:
                continue
            who = [i for i in range(n) if v in cand[i]]
            if len(who) == 1 and fixed[who[0]] is None and len(cand[who[0]]) > 1:
                i = who[0]
                lines.append(f"Only {names[i]} can be on floor {v}.")
                cand[i] = {v}
                fix(i, v)
                changed = True
        if all(f is not None for f in fixed):
            break
        if not changed:
            break
    if not all(f is not None for f in fixed):
        return {"solution": None, "lines": lines, "fingerprint": {"fail": "stall"}}
    # fingerprint: for deferred clues, how many fixes preceded their first application (chain position)
    defer_pos = sorted(
        first_applied[ci][1] for ci, c in enumerate(clues) if ci in first_applied and clue_pattern(c) in DEFERRED
    )
    fp = {
        "rounds": rounds,
        "n_lines": len(lines),
        "defer_fixes_before": defer_pos,
        "chain": len(set(defer_pos)),
        "unapplied": [ci for ci in range(len(clues)) if ci not in first_applied],
    }
    return {"solution": list(fixed), "lines": lines, "fingerprint": fp}


def oracle_trace(meta, answer, style=0, order_seed=0, unary="mixed"):
    names = [str(x) for x in meta["names"]]
    res = solve_verbal(
        names,
        list(meta["domain"]),
        list(meta["friends"]),
        [tuple(_tup(c)) for c in meta["clues"]],
        style,
        order_seed,
        unary,
    )
    if res["solution"] is None or list(res["solution"]) != list(meta["solution"]):
        return None, None
    return res["solution"], "\n".join(res["lines"]) + f"\n\\boxed{{{answer}}}"


def _tup(x):
    return tuple(_tup(y) for y in x) if isinstance(x, list) else x


if __name__ == "__main__":
    # smoke test: python -m nl_tower.oracle N SEED
    import sys

    from nl_tower import generator as gen

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cfg = gen.NLTowerConfig(n_people_min=n, n_people_max=n, min_deferred=1, size=4, seed=seed)
    ds = gen.NLTowerDataset(cfg)
    for i in range(4):
        e = ds[i]
        m = e["metadata"]
        print(e["question"])
        print("ANSWER", e["answer"])
        print("patterns", m["patterns"], "residual", m["residual_no_deferred"], "oracle", m["oracle"])
        sol, tr = oracle_trace(m, e["answer"])
        print(tr)
        print("score", ds.score_answer(e["answer"], e))
        print("-" * 80)
