"""Integrated possession predictor: WS2 rules team evidence + WS4 dead-ball classifier for the D column,
decoded by forward-backward posterior argmax (WS3, without the sampling) or by Viterbi.

    models = load_models()                       # possession/models/dead_v2.joblib
    params = load_params()                       # possession/params_v2.json
    codes  = predict(half, params, models)       # (T,) int8, H 0 / A 1 / D 2

`models` is {"model": fitted classifier, "keep": column indices, "features": names} or None (rules only,
D column from safety.emissions). Nothing here loads competition data; CPU, deterministic.
"""
import json
from pathlib import Path

import numpy as np

from . import dead, mbr, rules, safety
from .data import pm

HERE = Path(__file__).parent
BALL_PREFIXES = ("b_", "d_nearest", "n_near", "d_gap", "d_min_team")  # ball-derived dead features
RULES = {"R": 1.0, "BV": 0.5, "KW": 20, "W_HOLD": 0.3, "W_DIFF": 0.6}  # WS2 final, fixed here
LOG2 = np.log(2.0)


def keep_columns(names, variant):
    """Column indices of the dead features for variant 'players' (no ball features) or 'ball' (all)."""
    if variant == "ball":
        return np.arange(len(names))
    if variant != "players":
        raise ValueError(variant)
    return np.array([i for i, n in enumerate(names) if not n.startswith(BALL_PREFIXES)])


def base_params(params=None):
    base = json.loads((HERE / "params.json").read_text())
    return {**safety.PARAMS, **rules.PARAMS, **base, **RULES, **(params or {})}


def team_evidence(half, p, cache):
    """WS2 team evidence (T, 2) without the dead damping, plus the full rules evidence (T, 3)."""
    if "team" not in cache:
        e = rules.evidence(half, p, cache)
        damp = rules._team_damp(cache["f"], p)
        cache["e_rules"] = e
        cache["team"] = e[:, :pm.D] - damp[:, None]
    return cache["team"]


def dead_probability(half, models, cache):
    if "p_dead" not in cache:
        X, _ = dead.build_features(half)
        cache["p_dead"] = models["model"].predict_proba(X[:, models["keep"]])[:, 1]
    return cache["p_dead"]


def combine(team, p_dead, alpha):
    """Team evidence + D column from alpha * logit(p_dead), with the dead damping on the team columns."""
    z = np.clip(alpha * dead.logit(p_dead), -20, 20)
    e = np.empty((len(p_dead), 3))
    e[:, pm.D] = LOG2 - np.logaddexp(0, -z)
    e[:, :pm.D] = team + (LOG2 - np.logaddexp(0, z))[:, None]
    return e


def evidence(half, params, models, cache=None):
    p = base_params(params)
    cache = cache if cache is not None else {}
    team = team_evidence(half, p, cache)
    if models is None:
        return cache["e_rules"]
    return combine(team, dead_probability(half, models, cache), p["ALPHA"])


def decode(e, params):
    p = base_params(params)
    if p.get("DECODER", "argmax") == "viterbi":
        return safety.viterbi(e / p.get("TEMP", 1.0), p["SWITCH_TEAM"] / p.get("TEMP", 1.0),
                              p["SWITCH_DEAD"] / p.get("TEMP", 1.0))
    es = e / p["TEMP"]
    alpha, beta, logz = mbr.forward_backward(es, p["SWITCH_TEAM"], p["SWITCH_DEAD"])
    return np.argmax(mbr.marginals(alpha, beta, logz), axis=1).astype(np.int8)


def predict(half, params, models, cache=None):
    return decode(evidence(half, params, models, cache), params)


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v2.json").read_text())


def load_models(path=None):
    import joblib
    return joblib.load(path or HERE / "models" / "dead_v2.joblib")


def v3_predictor():
    """-> (params, predict(half)) for v3 (TCN seed ensemble + v2.1 evidence, 50/50, Viterbi). torch imported only here."""
    from . import tcn_v3
    p = tcn_v3.load_params()
    models, nets = load_models(HERE / "models" / "dead_v2_1.joblib"), tcn_v3.load_nets(params=p)
    return p, lambda half: tcn_v3.predict(half, p, models, nets)


def v4_predictor():
    """-> (params, predict(half)) for v4 (TCN4 seed ensemble + v2.1 evidence, 50/50, boundary-hook Viterbi). torch imported only here."""
    from . import tcn_v4
    p = tcn_v4.load_params()
    models, nets = load_models(HERE / "models" / "dead_v2_1.joblib"), tcn_v4.load_nets(params=p)
    return p, lambda half: tcn_v4.predict(half, p, models, nets)


def v10_predictor():
    """-> (params, predict(half)) for v10 (hedge: v6 + v8 nets in one ensemble). torch imported only here."""
    from . import tcn_v10
    p = tcn_v10.load_params()
    models, nets = load_models(HERE / "models" / "dead_v2_1.joblib"), tcn_v10.load_nets(params=p)
    return p, lambda half: tcn_v10.predict(half, p, models, nets)


def v8_predictor():
    """-> (params, predict(half)) for v8 (v6 recipe, nets trained on touch-merged labels). torch imported only here."""
    from . import tcn_v8
    p = tcn_v8.load_params()
    models, nets = load_models(HERE / "models" / "dead_v2_1.joblib"), tcn_v8.load_nets(params=p)
    return p, lambda half: tcn_v8.predict(half, p, models, nets)


def v6_predictor():
    """-> (params, predict(half)) for v6 (v4 recipe, nets trained with short-spell loss weight x3, v4 decoder). torch imported only here."""
    from . import tcn_v6
    p = tcn_v6.load_params()
    models, nets = load_models(HERE / "models" / "dead_v2_1.joblib"), tcn_v6.load_nets(params=p)
    return p, lambda half: tcn_v6.predict(half, p, models, nets)
