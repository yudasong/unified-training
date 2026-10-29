"""nl_tower: floor-assignment puzzles with referential clues (the reasoning puzzles of the paper, Appendix B).

n people live on distinct floors of a tower; the set of occupied floors is shown; every person has a stated best
friend (a fixed-point-free permutation, listed in the prompt).  Clues relate the FLOORS of two "terms".  A term is a
name, or a description whose referent is only known once part of the assignment is solved:

    ("n", i)           Ava
    ("friend", T)      Ava's best friend / the best friend of <T>
    ("occ", F)         the person on floor F
    ("rank", k)        the person on the highest / second-highest / lowest / second-lowest floor
                       (k indexes the sorted domain)
    ("above", T)       the person directly above <T>          ("below", T) likewise
Clue kinds: gt (higher floor), diff (exactly k floors above), sum, parity, mult (twice / three times), extreme.

Solving needs (a) deferring a clue, (b) resolving an identity from the partial solution, (c) a table lookup and
(d) the rule: "the best friend of the person on floor 19" can only be used once floor 19's occupant is fixed.
Midtraining data covers a sub-graph of these compositions; RL prompts use the full graph. Clue sets are minimised so
every clue (in particular every deferred one) is necessary for uniqueness, and every puzzle is accepted only if the
deterministic solver in oracle.py finishes it without branching (``require_oracle=True``; generation fails if the
solver cannot be imported). The answer is "Ava=12, Noah=9, ..." (exact dict match, ``score_answer``).

Built on reasoning-gym (Apache-2.0): ``ProceduralDataset`` below reproduces the part of reasoning-gym's dataset base
class this generator uses, so reasoning-gym is optional; ``register_with_reasoning_gym()`` registers the task as
"nl_tower" for ``reasoning_gym.create_dataset``. Up to 12 people per puzzle."""

import re
from dataclasses import dataclass, field
from random import Random
from typing import Any

NAMES = [
    "Ava",
    "Liam",
    "Maya",
    "Noah",
    "Zoe",
    "Ethan",
    "Mia",
    "Lucas",
    "Ella",
    "Owen",
    "Ruby",
    "Felix",
    "Nora",
    "Jasper",
    "Iris",
    "Hugo",
]

# term patterns (sub-graph axis). Each pattern is a generator of terms; "class" orders them by composition depth.
PATTERNS = ["direct", "friend", "friend2", "order", "rankn", "dfriend_occ", "dfriend_rank", "dfriend_above", "nested"]
STEP_COVERED = {"friend", "friend2", "order", "rankn"}
DEFERRED = {"dfriend_occ", "dfriend_rank", "dfriend_above", "nested"}  # identity of a floor's holder is needed


@dataclass
class NLTowerConfig:
    n_people_min: int = 6
    n_people_max: int = 7
    min_value: int = 1
    max_value: int = 40
    # pattern weights for the SUBJECT term of a clue (object terms are a name or a friend-of-name)
    patterns: dict = field(
        default_factory=lambda: {
            "direct": 4,
            "friend": 2,
            "friend2": 1,
            "order": 1,
            "dfriend_occ": 1,
            "dfriend_rank": 1,
            "dfriend_above": 1,
            "nested": 0,
        }
    )
    kinds: dict = field(default_factory=lambda: {"gt": 1, "diff": 3, "sum": 3, "parity": 1, "mult": 1, "extreme": 1})
    min_deferred: int = 0  # required number of NECESSARY deferred clues (after minimisation)
    max_deferred: int = 99
    n_redundant: int = 0  # extra true clues appended after minimisation
    require_oracle: bool = True  # keep only puzzles the verbal oracle solves without branching
    max_attempts: int = 400
    # chain mode: the puzzle is an explicit dependency chain. An anchor clue fixes 1-2 people; every
    # later clue's subject is a term that resolves only from what is fixed so far and pins one more person.
    mode: str = "chain"  # "chain" | "random"
    anchor: dict = field(default_factory=lambda: {"extreme": 1, "pair": 2})
    step_patterns: dict = field(
        default_factory=lambda: {
            "friend": 3,
            "friend2": 1,
            "order": 2,
            "rankn": 1,
            "dfriend_occ": 2,
            "dfriend_rank": 1,
            "dfriend_above": 1,
            "nested": 0,
        }
    )
    deferred_min: int = 1  # number of chain steps whose pattern is in req_patterns (the HELD-OUT compositions)
    deferred_max: int = 3
    req_patterns: tuple = ("dfriend_rank", "dfriend_above", "nested")
    n_noise: int = 0  # extra weak true clues (gt / parity between names) that do not pin anyone
    seed: int | None = None
    size: int = 500

    def validate(self):
        # at most 12 people (16 names); the domain check below also bounds n (3n <= 40)
        assert 4 <= self.n_people_min <= self.n_people_max <= 12
        assert self.max_value - self.min_value + 1 >= 3 * self.n_people_max
        assert all(p in PATTERNS for p in self.patterns)
        assert all(p in PATTERNS for p in self.step_patterns) and self.mode in ("chain", "random")


# ---------------------------------------------------------------- term / clue semantics
def rank_floor(domain, k):
    return domain[k]


def next_floor(domain, v, up=True):
    i = domain.index(v)
    if up:
        return domain[i + 1] if i + 1 < len(domain) else None
    return domain[i - 1] if i - 1 >= 0 else None


def term_name(t, fl, domain, fr, holder):
    """Name index the term refers to, given a (possibly partial) assignment fl (None = unknown), holder floor->idx.
    Returns an int, None (unknown yet) or False (undefined, e.g. 'above' the top floor)."""
    k = t[0]
    if k == "n":
        return t[1]
    if k == "friend":
        x = term_name(t[1], fl, domain, fr, holder)
        return x if x is None or x is False else fr[x]
    if k == "occ":
        return holder.get(t[1])
    if k == "rank":
        return holder.get(domain[t[1]])
    if k in ("above", "below"):
        x = term_name(t[1], fl, domain, fr, holder)
        if x is None or x is False:
            return x
        v = fl[x]
        if v is None:
            return None
        nv = next_floor(domain, v, up=(k == "above"))
        if nv is None:
            return False
        return holder.get(nv)
    raise ValueError(k)


def term_floor(t, fl, domain, fr, holder):
    """Floor value of the term (int), None (unknown yet) or False (undefined)."""
    k = t[0]
    if k == "occ":
        return t[1]
    if k == "rank":
        return domain[t[1]]
    if k in ("above", "below"):
        x = term_floor(t[1], fl, domain, fr, holder)
        if x is None or x is False:
            return x
        nv = next_floor(domain, x, up=(k == "above"))
        return False if nv is None else nv
    x = term_name(t, fl, domain, fr, holder)
    if x is None or x is False:
        return x
    return fl[x]


def clue_check(c, fl, domain, fr, holder):
    """True / False / None(unknown yet)."""
    kind = c[0]
    a = term_floor(c[1], fl, domain, fr, holder)
    if a is False:
        return False
    if kind == "parity":
        return None if a is None else (a % 2 == c[3])
    if kind == "extreme":
        return None if a is None else (a == (domain[-1] if c[3] == "high" else domain[0]))
    if kind == "rankc":
        return None if a is None else (a == domain[c[3]])
    b = term_floor(c[2], fl, domain, fr, holder)
    if b is False:
        return False
    if a is None or b is None:
        return None
    if kind == "gt":
        return a > b
    if kind == "diff":
        return a - b == c[3]
    if kind == "sum":
        return a + b == c[3]
    if kind == "mult":
        return a == c[3] * b
    raise ValueError(kind)


def count_solutions(n, domain, fr, clues, limit=2, want=False):
    """Backtracking count of assignments (names -> distinct floors) satisfying all clues; stops at `limit`."""
    fl = [None] * n
    holder = {}
    used = set()
    found = []

    def rec(i):
        if i == n:
            if all(clue_check(c, fl, domain, fr, holder) is True for c in clues):
                if want:
                    found.append(list(fl))
                return 1
            return 0
        tot = 0
        for v in domain:
            if v in used:
                continue
            fl[i] = v
            holder[v] = i
            used.add(v)
            if all(clue_check(c, fl, domain, fr, holder) is not False for c in clues):
                tot += rec(i + 1)
            fl[i] = None
            del holder[v]
            used.discard(v)
            if tot >= limit:
                break
        return tot

    cnt = rec(0)
    return (cnt, found) if want else cnt


# ---------------------------------------------------------------- text
RANK_WORD = {}


def rank_word(domain, k):
    n = len(domain)
    return {n - 1: "highest", n - 2: "second-highest", 0: "lowest", 1: "second-lowest"}[k]


def term_text(t, names, domain):
    k = t[0]
    if k == "n":
        return names[t[1]]
    if k == "friend":
        if t[1][0] == "n":
            return f"{names[t[1][1]]}'s best friend"
        return f"the best friend of {term_text(t[1], names, domain)}"
    if k == "occ":
        return f"the person on floor {t[1]}"
    if k == "rank":
        return f"the person on the {rank_word(domain, t[1])} floor"
    if k == "above":
        return f"the person directly above {term_text(t[1], names, domain)}"
    if k == "below":
        return f"the person directly below {term_text(t[1], names, domain)}"
    raise ValueError(k)


def cap(s):
    return s[0].upper() + s[1:]


def clue_text(c, names, domain):
    kind = c[0]
    t1 = term_text(c[1], names, domain)
    if kind == "parity":
        return cap(f"{t1} lives on an {'odd' if c[3] else 'even'} floor.")
    if kind == "extreme":
        return cap(f"{t1} lives on the {'highest' if c[3] == 'high' else 'lowest'} floor.")
    if kind == "rankc":
        return cap(f"{t1} lives on the {rank_word(domain, c[3])} floor.")
    t2 = term_text(c[2], names, domain)
    if kind == "gt":
        return cap(f"{t1} lives on a higher floor than {t2}.")
    if kind == "diff":
        return cap(f"{t1} lives {c[3]} floor{'s' if c[3] != 1 else ''} above {t2}.")
    if kind == "sum":
        return cap(f"the floor numbers of {t1} and {t2} add up to {c[3]}.")
    if kind == "mult":
        return cap(f"the floor number of {t1} is {'twice' if c[3] == 2 else 'three times'} that of {t2}.")
    raise ValueError(kind)


def term_pattern(t):
    """Pattern class of a term (for the sub-graph bookkeeping)."""
    k = t[0]
    if k == "n":
        return "direct"
    if k == "friend":
        inner = t[1][0]
        if inner == "n":
            return "friend"
        if inner == "friend" and t[1][1][0] == "n":
            return "friend2"
        if inner == "occ":
            return "dfriend_occ"
        if inner == "rank":
            return "dfriend_rank"
        if inner in ("above", "below") and t[1][1][0] == "n":
            return "dfriend_above"
        return "nested"
    if k in ("above", "below"):
        return "order" if t[1][0] == "n" else "nested"
    if k == "rank":
        return "rankn"  # bare rank -> a floor constant (numeric anchor, covered primitive)
    if k == "occ":
        return "const"  # bare floor constant (never generated)
    raise ValueError(k)


def term_depth(t):
    return 0 if t[0] == "n" else (1 if t[0] in ("occ", "rank") else 1 + term_depth(t[1]))


PORDER = {
    "direct": 0,
    "const": 0,
    "friend": 1,
    "friend2": 2,
    "order": 2,
    "rankn": 2,
    "dfriend_occ": 3,
    "dfriend_rank": 3,
    "dfriend_above": 4,
    "nested": 5,
}


def clue_pattern(c):
    ps = [term_pattern(c[1])] + ([term_pattern(c[2])] if c[0] in ("gt", "diff", "sum", "mult") else [])
    if c[0] == "rankc":
        ps.append("rankn")
    return max(ps, key=lambda p: PORDER[p])


# ---------------------------------------------------------------- generation
def _wchoice(rng, weights):
    items = [(k, w) for k, w in weights.items() if w > 0]
    tot = sum(w for _, w in items)
    r = rng.random() * tot
    for k, w in items:
        r -= w
        if r <= 0:
            return k
    return items[-1][0]


def _sample_term(rng, pattern, n, domain, fl):
    i = rng.randrange(n)
    if pattern == "direct":
        return ("n", i)
    if pattern == "friend":
        return ("friend", ("n", i))
    if pattern == "friend2":
        return ("friend", ("friend", ("n", i)))
    if pattern == "order":
        return (rng.choice(["above", "below"]), ("n", i))
    if pattern == "dfriend_occ":
        return ("friend", ("occ", rng.choice(domain)))
    if pattern == "dfriend_rank":
        return ("friend", ("rank", rng.choice([n - 1, n - 2, 0, 1])))
    if pattern == "dfriend_above":
        return ("friend", (rng.choice(["above", "below"]), ("n", i)))
    if pattern == "nested":
        j = rng.randrange(n)  # noqa: F841 - unused, but the draw advances rng; removing it changes the puzzles
        return rng.choice(
            [
                ("friend", (rng.choice(["above", "below"]), ("friend", ("n", i)))),
                ("friend", ("friend", ("occ", rng.choice(domain)))),
                ("friend", (rng.choice(["above", "below"]), ("occ", rng.choice(domain)))),
                ("friend", ("friend", ("rank", rng.choice([n - 1, 0])))),
            ]
        )
    raise ValueError(pattern)


def _sample_clue(rng, cfg, n, domain, fl, fr, holder):
    kind = _wchoice(rng, cfg.kinds)
    t1 = _sample_term(rng, _wchoice(rng, cfg.patterns), n, domain, fl)
    a = term_floor(t1, fl, domain, fr, holder)
    if a is None or a is False:
        return None
    if kind == "parity":
        if t1[0] in ("occ", "rank"):
            return None
        return ("parity", t1, None, a % 2)
    if kind == "extreme":
        if t1[0] in ("occ", "rank"):
            return None
        if a == domain[-1]:
            return ("extreme", t1, None, "high")
        if a == domain[0]:
            return ("extreme", t1, None, "low")
        return None
    # object term: mostly a plain name, sometimes a friend-of-name (keeps sentences readable)
    t2 = _sample_term(rng, "direct" if rng.random() < 0.75 else "friend", n, domain, fl)
    b = term_floor(t2, fl, domain, fr, holder)
    if b is None or b is False or a == b:
        return None
    if kind == "gt":
        return ("gt", t1, t2, None) if a > b else ("gt", t2, t1, None)
    if kind == "diff":
        return ("diff", t1, t2, a - b) if a > b else ("diff", t2, t1, b - a)
    if kind == "sum":
        return ("sum", t1, t2, a + b)
    if kind == "mult":
        for m in (2, 3):
            if a == m * b:
                return ("mult", t1, t2, m)
            if b == m * a:
                return ("mult", t2, t1, m)
        return None
    raise ValueError(kind)


def _clue_key(c):
    return repr(c)


# ---------------------------------------------------------------- chain construction (v2)
def _unique_diff_pairs(domain):
    """ordered (a,b) with a>b whose difference occurs once in the domain."""
    from collections import Counter

    d = Counter(a - b for a in domain for b in domain if a > b)
    return {(a, b) for a in domain for b in domain if a > b and d[a - b] == 1}


def _unique_sum_pairs(domain):
    from collections import Counter

    d = Counter(a + b for i, a in enumerate(domain) for b in domain[i + 1 :])
    return {(a, b) for i, a in enumerate(domain) for b in domain[i + 1 :] if d[a + b] == 1}


def _npairs(domain, kind, arg):
    if kind == "diff":
        return sum(1 for a in domain for b in domain if a - b == arg)
    if kind == "sum":
        return sum(1 for a in domain for b in domain if a != b and a + b == arg)
    if kind == "mult":
        return sum(1 for a in domain for b in domain if a == arg * b)
    return 0


def _pin_clue(rng, term, y, z, fl, domain=None, weak=True):
    """A clue with subject term `term` (which resolves to y) and object name z (already fixed) that pins y.
    weak=True: only relations whose pair set in the domain has >= 2 members, so the clue pins y ONLY once z is
    known."""
    a, b = fl[y], fl[z]
    opts = []
    if not weak or _npairs(domain, "diff", abs(a - b)) >= 2:
        opts.append(("diff", 2))
    if not weak or _npairs(domain, "sum", a + b) >= 2:
        opts.append(("sum", 2))
    if a in (2 * b, 3 * b) or b in (2 * a, 3 * a):
        m = a // b if a > b else b // a
        if not weak or _npairs(domain, "mult", m) >= 2:
            opts.append(("mult", 2))
    if not opts:
        return None
    kind = _wchoice(rng, dict(opts))
    if kind == "sum":
        return ("sum", term, ("n", z), a + b)
    if kind == "mult":
        if a in (2 * b, 3 * b):
            return ("mult", term, ("n", z), a // b)
        return ("mult", ("n", z), term, b // a)
    return ("diff", term, ("n", z), a - b) if a > b else ("diff", ("n", z), term, b - a)


def _pin_numeric(rng, term, y, fl, const):
    """Subject `term` is a floor constant (order / rankn); pin the name y relative to it."""
    a = fl[y]
    if rng.random() < 0.6 or a + const > 2 * max(fl):
        return ("diff", ("n", y), term, a - const) if a > const else ("diff", term, ("n", y), const - a)
    return ("sum", ("n", y), term, a + const)


def gen_chain(rng, cfg, n, domain, fl, fr, holder):
    """Returns (clues, n_deferred) or None."""
    unfixed = set(range(n))
    fixedl = []
    clues = []
    # anchor
    if _wchoice(rng, cfg.anchor) == "extreme":
        k = rng.choice([n - 1, 0, n - 2, 1])
        i = holder[domain[k]]
        if k in (n - 1, 0):
            clues.append(("extreme", ("n", i), None, "high" if k == n - 1 else "low"))
        else:
            clues.append(("rankc", ("n", i), None, k))
        fixedl.append(i)
        unfixed.discard(i)
    else:
        cands = [(holder[a], holder[b]) for (a, b) in _unique_diff_pairs(domain)]
        scands = [(holder[a], holder[b]) for (a, b) in _unique_sum_pairs(domain)]
        if cands and (not scands or rng.random() < 0.6):
            i, j = rng.choice(sorted(cands))
            clues.append(("diff", ("n", i), ("n", j), fl[i] - fl[j]))
        elif scands:
            i, j = rng.choice(sorted(scands))
            clues.append(("sum", ("n", i), ("n", j), fl[i] + fl[j]))
        else:
            return None
        for x in (i, j):
            fixedl.append(x)
            unfixed.discard(x)
    n_def = 0
    req = set(cfg.req_patterns)
    want_def = rng.randint(cfg.deferred_min, cfg.deferred_max)
    steps_left = n - len(fixedl) - 1  # last person falls out by elimination
    if want_def > steps_left:
        return None
    while steps_left > 0:
        # decide step class: deferred steps are spread over the remaining positions
        need_def = want_def - n_def
        force_def = need_def >= steps_left
        pw = {
            k: w
            for k, w in cfg.step_patterns.items()
            if w > 0 and ((k in req) if force_def else (need_def > 0 or k not in req))
        }
        if not pw:
            return None
        made = None
        for _try in range(30):
            pat = _wchoice(rng, pw)
            z = (
                fixedl[-1] if rng.random() < 0.8 else rng.choice(fixedl[-2:])
            )  # object: the person fixed last (linear chain)
            fx = set(fixedl)
            if pat == "friend":
                x = rng.randrange(n)
                y = fr[x]
                if y not in unfixed:
                    continue
                made = (_pin_clue(rng, ("friend", ("n", x)), y, z, fl, domain), y)
            elif pat == "friend2":
                x = rng.randrange(n)
                y = fr[fr[x]]
                if y not in unfixed:
                    continue
                made = (_pin_clue(rng, ("friend", ("friend", ("n", x))), y, z, fl, domain), y)
            elif pat == "order":
                x = fixedl[-1]
                up = rng.random() < 0.5
                f = next_floor(domain, fl[x], up=up)
                if f is None:
                    continue
                y = rng.choice(sorted(unfixed))
                made = (_pin_numeric(rng, ("above" if up else "below", ("n", x)), y, fl, f), y)
            elif pat == "rankn":
                k = rng.choice([n - 1, n - 2, 0, 1])
                y = rng.choice(sorted(unfixed))
                if fl[y] == domain[k]:
                    continue
                made = (_pin_numeric(rng, ("rank", k), y, fl, domain[k]), y)
            elif pat == "dfriend_occ":
                x = rng.choice(fixedl)
                y = fr[x]
                if y not in unfixed:
                    continue
                made = (_pin_clue(rng, ("friend", ("occ", fl[x])), y, z, fl, domain), y)
            elif pat == "dfriend_rank":
                ks = [k for k in (n - 1, n - 2, 0, 1) if holder[domain[k]] in fx and fr[holder[domain[k]]] in unfixed]
                if not ks:
                    continue
                k = rng.choice(ks)
                y = fr[holder[domain[k]]]
                made = (_pin_clue(rng, ("friend", ("rank", k)), y, z, fl, domain), y)
            elif pat == "dfriend_above":
                opts = []
                for x in fixedl:
                    for up in (True, False):
                        f = next_floor(domain, fl[x], up=up)
                        if f is not None and holder[f] in fx and fr[holder[f]] in unfixed:
                            opts.append((x, up))
                if not opts:
                    continue
                x, up = rng.choice(opts)
                y = fr[holder[next_floor(domain, fl[x], up=up)]]
                made = (_pin_clue(rng, ("friend", ("above" if up else "below", ("n", x))), y, z, fl, domain), y)
            elif pat == "nested":
                opts = []
                for x in range(n):  # friend(above(friend(x)))
                    w = fr[x]
                    if w in fx:
                        for up in (True, False):
                            f = next_floor(domain, fl[w], up=up)
                            if f is not None and holder[f] in fx and fr[holder[f]] in unfixed:
                                opts.append(
                                    (("friend", ("above" if up else "below", ("friend", ("n", x)))), fr[holder[f]])
                                )
                for x in fixedl:  # friend(friend(occ(F)))
                    if fr[fr[x]] in unfixed:
                        opts.append((("friend", ("friend", ("occ", fl[x]))), fr[fr[x]]))
                    for up in (True, False):  # friend(above(occ(F)))
                        f = next_floor(domain, fl[x], up=up)
                        if f is not None and holder[f] in fx and fr[holder[f]] in unfixed:
                            opts.append((("friend", ("above" if up else "below", ("occ", fl[x]))), fr[holder[f]]))
                for k in (n - 1, 0):  # friend(friend(rank))
                    if holder[domain[k]] in fx and fr[fr[holder[domain[k]]]] in unfixed:
                        opts.append((("friend", ("friend", ("rank", k))), fr[fr[holder[domain[k]]]]))
                if not opts:
                    continue
                term, y = rng.choice(opts)
                made = (_pin_clue(rng, term, y, z, fl, domain), y)
            if made is not None and made[0] is None:
                made = None
            if made is not None:
                break
        if made is None:
            return None
        c, y = made
        if clue_pattern(c) in req:
            n_def += 1
        clues.append(c)
        fixedl.append(y)
        unfixed.discard(y)
        steps_left -= 1
    if n_def != want_def:
        return None
    # weak noise clues on plain names
    for _ in range(60):
        if sum(1 for c in clues if c[0] in ("gt", "parity") and c[1][0] == "n") >= cfg.n_noise:
            break
        i = rng.randrange(n)
        if rng.random() < 0.5:
            c = ("parity", ("n", i), None, fl[i] % 2)
        else:
            j = rng.randrange(n)
            if i == j:
                continue
            c = ("gt", ("n", i), ("n", j), None) if fl[i] > fl[j] else ("gt", ("n", j), ("n", i), None)
        if _clue_key(c) not in {_clue_key(x) for x in clues}:
            clues.append(c)
    return clues, n_def


class ProceduralDataset:
    """reasoning-gym's ProceduralDataset semantics without the dependency.

    The config is validated, ``seed`` defaults to a random draw when None, ``len()`` is ``size`` and iteration yields
    ``self[0] .. self[size - 1]``. NLTowerDataset items are pure functions of ``(seed, idx)`` (``Random(seed + idx)``),
    so a dataset built here equals one built by ``reasoning_gym.create_dataset("nl_tower", ...)``."""

    def __init__(self, config: Any, seed: int | None = None, size: int = 500):
        if hasattr(config, "validate") and callable(config.validate):
            config.validate()
        self.config = config
        self.size = size
        self.seed = seed if seed is not None else Random().randint(0, 2**32)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self) -> dict[str, Any]:
        if self._current_idx >= self.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError


class NLTowerDataset(ProceduralDataset):
    def __init__(self, config: NLTowerConfig, seed=None, size=None):
        if seed is None:
            seed = config.seed
        if size is None:
            size = config.size
        super().__init__(config=config, seed=seed, size=size)

    def _oracle_ok(self, names, domain, fr, clues, sol):
        if not self.config.require_oracle:
            return True, {}
        # No fallback: a missing solver must stop generation (accepting every puzzle would change the sets).
        try:
            from .oracle import solve_verbal
        except ImportError as e:
            raise ImportError(
                "nl_tower: the solver (nl_tower.oracle) cannot be imported; refusing to generate "
                "puzzles without the oracle acceptance filter (require_oracle=True)"
            ) from e
        res = solve_verbal(names, domain, fr, clues, style=0, order_seed=0)
        ok = res["solution"] is not None and list(res["solution"]) == list(sol)
        return ok, res.get("fingerprint", {})

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        cfg = self.config
        for _attempt in range(cfg.max_attempts):
            n = rng.randint(cfg.n_people_min, cfg.n_people_max)
            names = rng.sample(NAMES, n)
            domain = sorted(rng.sample(range(cfg.min_value, cfg.max_value + 1), n))
            fl = list(domain)
            rng.shuffle(fl)
            holder = {v: i for i, v in enumerate(fl)}
            # fixed-point-free friend permutation
            while True:
                fr = list(range(n))
                rng.shuffle(fr)
                if all(fr[i] != i for i in range(n)):
                    break
            if cfg.mode == "chain":
                g = gen_chain(rng, cfg, n, domain, fl, fr, holder)
                if g is None:
                    continue
                clues, nd = g
                if not all(clue_check(c, fl, domain, fr, holder) is True for c in clues):
                    continue
                if count_solutions(n, domain, fr, clues, limit=2) != 1:
                    continue
                keep = clues
                rng.shuffle(clues)
                ok, fp = self._oracle_ok(names, domain, fr, clues, fl)
                if not ok:
                    continue
                break
            # 1) add true clues until the solution is unique
            chosen, seen = [], set()
            for _ in range(400):
                c = _sample_clue(rng, cfg, n, domain, fl, fr, holder)
                if c is None or _clue_key(c) in seen:
                    continue
                if clue_check(c, fl, domain, fr, holder) is not True:
                    continue
                cnt = count_solutions(n, domain, fr, chosen + [c], limit=2)
                if cnt < 1:
                    continue  # should not happen (true clue), keep safe
                # keep only clues that cut something (cheap test: still >=2 solutions without it, or unique now)
                seen.add(_clue_key(c))
                chosen.append(c)
                if cnt == 1:
                    break
            else:
                continue
            if count_solutions(n, domain, fr, chosen, limit=2) != 1:
                continue
            # 2) minimise: drop clues whose removal keeps uniqueness
            #    (random order; deferred clues tried LAST so they survive)
            order = list(range(len(chosen)))
            rng.shuffle(order)
            order.sort(key=lambda i: clue_pattern(chosen[i]) in DEFERRED)  # non-deferred first -> removed first
            keep = list(chosen)
            for i in order:
                c = chosen[i]
                trial = [x for x in keep if x is not c]
                if count_solutions(n, domain, fr, trial, limit=2) == 1:
                    keep = trial
            nd = sum(1 for c in keep if clue_pattern(c) in DEFERRED)
            if not (cfg.min_deferred <= nd <= cfg.max_deferred):
                continue
            # 3) optional redundant clues (true, not already present)
            extra = []
            for _ in range(200):
                if len(extra) >= cfg.n_redundant:
                    break
                c = _sample_clue(rng, cfg, n, domain, fl, fr, holder)
                if c is None or clue_check(c, fl, domain, fr, holder) is not True:
                    continue
                if _clue_key(c) in {_clue_key(x) for x in keep + extra}:
                    continue
                extra.append(c)
            clues = keep + extra
            rng.shuffle(clues)
            ok, fp = self._oracle_ok(names, domain, fr, clues, fl)
            if not ok:
                continue
            break
        else:
            raise RuntimeError("generation failed")
        # residual ambiguity when every deferred clue is ignored (how much a solver that skips them could guess)
        nodefer = [c for c in clues if clue_pattern(c) not in DEFERRED]
        residual = count_solutions(n, domain, fr, nodefer, limit=5000)
        noheld = [c for c in clues if clue_pattern(c) not in set(cfg.req_patterns)]
        residual_h = count_solutions(n, domain, fr, noheld, limit=5000)
        stmts = [clue_text(c, names, domain) for c in clues]
        friends = ", ".join(f"{names[i]}'s best friend is {names[fr[i]]}" for i in range(n))
        q = (
            f"{', '.join(names[:-1])} and {names[-1]} live on different floors of the same tower. "
            f"The occupied floors are {{{', '.join(map(str, domain))}}}. "
            f"Best friends: {friends}. Here is what is known:\n"
            + "\n".join(f" - {s}" for s in stmts)
            + "\nWork out which floor each person lives on. Answer with a comma-separated list like "
            + f'"{names[0]}=NUMBER, ..." covering everyone.'
        )
        answer = ", ".join(f"{names[i]}={fl[i]}" for i in range(n))
        pats = [clue_pattern(c) for c in clues]
        return {
            "question": q,
            "answer": answer,
            "metadata": {
                "source_dataset": "nl_tower",
                "source_index": idx,
                "names": names,
                "domain": domain,
                "friends": fr,
                "solution": fl,
                "clues": clues,
                "statements": stmts,
                "patterns": pats,
                "n_deferred": sum(1 for p in pats if p in DEFERRED),
                "n_held": sum(1 for p in pats if p in set(cfg.req_patterns)),
                "residual_no_held": residual_h,
                "n_necessary": len(keep),
                "max_depth": max(term_depth(c[1]) for c in clues),
                "residual_no_deferred": residual,
                "oracle": fp,
            },
        }

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        gold = dict(re.findall(r"(\w+)=(-?\d+)", entry["answer"]))
        got = dict(re.findall(r"(\w+)\s*=\s*(-?\d+)", str(answer)))
        return 1.0 if got == gold else 0.0


def register_with_reasoning_gym(name: str = "nl_tower") -> None:
    """Opt-in: register NLTowerDataset with an installed reasoning-gym under ``name`` (idempotent).

    reasoning-gym only accepts subclasses of its own base class, so the registered class combines both bases. A copy
    already registered under ``name`` (e.g. by a vendored reasoning-gym that ships custom/nl_tower.py) is replaced:
    this module is the reference."""
    from reasoning_gym.dataset import ProceduralDataset as RGProceduralDataset
    from reasoning_gym.factory import DATASETS, register_dataset

    class RGNLTowerDataset(NLTowerDataset, RGProceduralDataset):
        """NLTowerDataset as a reasoning-gym dataset."""

    if name in DATASETS:
        DATASETS[name] = (RGNLTowerDataset, NLTowerConfig)
    else:
        register_dataset(name, RGNLTowerDataset, NLTowerConfig)
