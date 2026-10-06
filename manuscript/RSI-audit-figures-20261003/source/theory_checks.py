#!/usr/bin/env python3
"""Finite numerical checks for the supplied 21-page RSI manuscript.

New local companion implementation, not a recovered author file. NumPy is the
only third-party dependency. All checks execute under python -O as well.
Numerical fixtures check implementation consistency; they do not prove theorems,
certify a neural-network smoothness constant, or supply LLM measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PAPER_SHA256 = "8efd10ce126a2ceeb58b8b498b6035fbb39a67bb504cf943870cf148d8e75883"
AB = ((.75, .5), (.6, .2), (.9, .7), (.4, .1))


class CheckFailure(Exception):
    pass


class Checks:
    def __init__(self):
        self.count = 0
        self.max_abs_error = 0.0

    def require(self, condition, label):
        self.count += 1
        if not bool(condition):
            raise CheckFailure(label)

    def close(self, actual, expected, label, atol=2e-12, rtol=2e-11):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        self.require(actual.shape == expected.shape, label + ": shape")
        self.require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),
                     label + ": finite")
        error = float(np.max(np.abs(actual - expected)))
        self.max_abs_error = max(self.max_abs_error, error)
        self.require(np.all(np.abs(actual - expected) <= atol + rtol * np.abs(expected)),
                     f"{label}: actual={actual}, expected={expected}")


def sigmoid(z):
    """Compute probability and complement without cancellation in the tails."""
    e = math.exp(-abs(float(z)))
    return (1 / (1 + e), e / (1 + e)) if z >= 0 else (e / (1 + e), 1 / (1 + e))


def enumeration(theta, a, b, L, rule="S", q=0.0):
    """Independent finite-outcome oracle: groups x signs x emitted labels.

    Allocate the S error quota greedily by group. Differentiate each candidate's
    BCE only. Audit queries include correct candidates; removal includes errors
    only. No functions from rsi.logistic or the closed-form audit field are used.
    """
    ps = [sigmoid(theta), sigmoid(L * theta)]
    rho = [1 - b, b]
    error_mass = [rho[j] * ps[j][1] for j in range(2)]
    if rule == "R":
        rates = [b, b]
    elif rule == "S":
        remaining = b * sum(error_mass)
        rates = [0.0, 0.0]
        for j in (1, 0):
            take = min(remaining, error_mass[j])
            rates[j] = take / error_mass[j] if error_mass[j] else 0.0
            remaining = max(0.0, remaining - take)
    else:
        raise ValueError(rule)
    rows = []
    for j, s in enumerate((1.0, L)):
        for sign in (-1, 1):
            x = sign * s
            truth = int(sign > 0)
            py1, py0 = sigmoid(theta * x)
            for y in (0, 1):
                correct = int(y == truth)
                prob = rho[j] * .5 * (py1 if y else py0)
                accepted = a if correct else rates[j]
                retained = accepted * (1 - q if j == 1 and not correct else 1)
                # Stable BCE derivative: (sigma(theta*x)-y)*x.
                grad = (py1 if y == 0 else -py0) * x
                rows.append(dict(prob=prob, correct=correct, high=j == 1,
                                 accepted=accepted, retained=retained, grad=grad))
    total = sum(r["prob"] for r in rows)
    pre = sum(r["prob"] * r["accepted"] for r in rows)
    z = sum(r["prob"] * r["retained"] for r in rows)
    numerator = sum(r["prob"] * r["retained"] * r["grad"] for r in rows)
    queries = sum(r["prob"] * r["accepted"] * q for r in rows if r["high"])
    error = sum(r["prob"] for r in rows if not r["correct"])
    return dict(rows=rows, total=total, rates=rates, z=z, pre=pre,
                gradient=numerator / z, numerator=numerator,
                accuracy=sum(r["prob"] for r in rows if r["correct"]),
                fpr=sum(r["prob"] * r["retained"] for r in rows if not r["correct"]) / error,
                query_fraction=queries / pre)


def piecewise_rates(theta, b, L):
    p, e = sigmoid(theta)
    pL, eL = sigmoid(L * theta)
    D = (1 - b) * e + b * eL
    return (0., D / eL) if theta <= 0 else (b * (pL - p) / e, 1.)


def audited_field(theta, a, b, L, q):
    """Equation (25), standalone population extension, not the GPU audit policy."""
    p, e = sigmoid(theta)
    pL, eL = sigmoid(L * theta)
    r1, rL = piecewise_rates(theta, b, L)
    P = (1 - b) * p + b * pL
    z = a * P + (1 - b) * e * r1 + b * eL * (1 - q) * rL
    n = (1 - b) * p * e * (r1 - a) + b * L * pL * eL * ((1 - q) * rL - a)
    return dict(gradient=n / z, numerator=n, z=z)


def check_general(c, m):
    cases = 0
    for a, b in AB:
        for L in (1., 1.25, 2., 3., 4., 10., 25.):
            limit = min(2., 18. / L)
            for theta in (-limit, -.1, -1e-8, 0., 1e-8, .1, limit):
                for rule in ("R", "S"):
                    label = f"a={a},b={b},L={L},theta={theta},rule={rule}"
                    e = enumeration(theta, a, b, L, rule)
                    c.close(e["total"], 1, label + ": probability")
                    c.close(m.acceptance_rates(theta, rule, a=a, b=b, L=L), e["rates"], label + ": acceptance")
                    c.close(m.stopped_gradient(theta, rule, a=a, b=b, L=L), e["gradient"], label + ": gradient")
                    c.close(m.sampling_accuracy(theta, b=b, L=L), e["accuracy"], label + ": accuracy")
                    c.close(m.acceptance_rate(theta, rule, a=a, b=b, L=L), e["z"], label + ": yield")
                    c.close(e["z"], a * e["accuracy"] + b * (1 - e["accuracy"]), label + ": matched Z")
                    c.close(e["fpr"], b, label + ": FPR")
                    if rule == "S":
                        c.close(e["rates"], piecewise_rates(theta, b, L), label + ": piecewise")
                    cases += 1
    return {"cases": cases, "L": [1, 1.25, 2, 3, 4, 10, 25],
            "scope": "Four (a,b) pairs; both sides of theta=0; bounded nonsaturated parameter grid."}


def check_initial(c, m):
    table = []
    for a, b in AB:
        for L in (1., 2., 3., 4., 10., 25.):
            sbar = 1 - b + b * L
            expected = ((b - a) * sbar / (2 * (a + b)),
                        (b * L - a * sbar) / (2 * (a + b)))
            gradients = [m.stopped_gradient(0, r, a=a, b=b, L=L) for r in ("R", "S")]
            c.close(gradients, expected, "Eq9 initial gradients")
            if (a, b) == (.75, .5) and L in (1, 2, 3, 10):
                fixture = m.fixed_128_fixture(L=L)
                for rule, g in zip(("R", "S"), gradients):
                    c.require(fixture["summary"][rule] == {"correct": 48, "error": 32}, "named-group 48+32 fixture")
                    c.close(m.sampling_accuracy(-.1 * g, b=b, L=L),
                            enumeration(-.1 * g, a, b, L, rule)["accuracy"], "one-step accuracy")
                if L == 1:
                    c.require([r["S_keep_error"] for r in fixture["points"]] == [0, 0, 16, 16],
                              "L=1 retains group identities")
                table.append(dict(L=int(L), g_R=gradients[0], g_S=gradients[1]))
    return {"table4": table, "fixture": "48 correct + 32 errors; separate from LLM K64 48+16"}


def check_covariance(c, m):
    # Use unequal correctness probabilities and genuinely vector-valued gradients.
    # Each rule has matched class-conditional acceptance, but both covariances
    # contribute. This is not the special case of identical correct responses.
    w = np.array([.08, .12, .20, .10, .15, .35])
    correct = np.array([True, True, True, False, False, False])
    G = np.array([[2, -1, 3], [-2, 4, 0], [1, 2, -3], [3, 2, 1], [-1, 3, 5], [2, -4, 1]], float)
    ar = np.where(correct, .75, .3)
    diff = np.array([.10, -.10 * .08 / .12, 0, .12, -.12 * .10 / .15, 0])
    ac = ar + diff
    c.require(np.all((ac >= 0) & (ac <= 1)), "valid acceptance probabilities")
    z = w @ ar
    lhs = ((w * ac) @ G - (w * ar) @ G) / z
    rhs = np.zeros(3)
    terms = []
    for mask in (correct, ~correct):
        mass = w[mask].sum()
        conditional = w[mask] / mass
        c.close(conditional @ diff[mask], 0, "matched conditional rates")
        covariance = (conditional * diff[mask]) @ G[mask] - (conditional @ diff[mask]) * (conditional @ G[mask])
        term = mass * covariance / z
        c.require(np.linalg.norm(term) > 1e-3, "nonzero covariance fixture")
        rhs += term
        terms.append(term.tolist())
    c.close(lhs, rhs, "Lemma3.1 covariance identity")
    c.require(np.linalg.norm(lhs) > 1e-3, "rates do not identify the vector gradient")
    return {"lhs": lhs.tolist(), "rhs": rhs.tolist(), "conditional_terms": terms}


def check_auditing(c, m):
    cases = 0
    max_denominator_trap = 0.
    for a, b in AB:
        for L in (1.25, 2., 4., 10., 25.):
            for theta in (-1., -.1, 0., .1, 1.):
                for q in (0., (1 - a) / 2, 1 - a, (2 - a) / 2, 1.):
                    e = enumeration(theta, a, b, L, "S", q)
                    f = audited_field(theta, a, b, L, q)
                    c.close([f["gradient"], f["z"], f["numerator"]],
                            [e["gradient"], e["z"], e["numerator"]], "Eq25 post-audit enumeration")
                    c.require(f["z"] > 0, "positive post-audit yield")
                    if q >= 1 - a:
                        c.require(f["gradient"] < 0, "uniform sufficient threshold: strictly improving")
                    if theta == 0:
                        g0 = (b * L * (1 - a - q) - a * (1 - b)) / (2 * (a + b * (1 - q)))
                        c.close(f["gradient"], g0, "Eq26 audited initial gradient")
                        c.close(e["query_fraction"], q * b * (a + 1) / (a + b), "Cor4.6 query fraction")
                    if q > 0:
                        max_denominator_trap = max(max_denominator_trap,
                            abs(e["numerator"] / e["pre"] - f["gradient"]))
                    cases += 1
    c.require(max_denominator_trap > .01, "fixture detects using the pre-audit denominator")
    contrasts = []
    for a, b in AB:
        for q in (0., .5 * (1 - a), .9 * (1 - a)):
            threshold = a * (1 - b) / (b * (1 - a - q))
            for factor, sign in ((.9, -1), (1., 0), (1.1, 1)):
                L = threshold * factor
                g = audited_field(0, a, b, L, q)["gradient"]
                if sign:
                    c.require(sign * g > 0, "contrast threshold correct side")
                else:
                    c.close(g, 0, "equality at contrast threshold")
                    theta = 0.
                    for _ in range(100):
                        theta -= .01 * audited_field(theta, a, b, L, q)["gradient"]
                    c.close(theta, 0, "equality: constant Euler trajectory", atol=2e-12)
                contrasts.append(dict(a=a,b=b,q=q,L=L,gradient=g))
    return {"enumeration_cases": cases, "contrast_cases": len(contrasts),
            "old_denominator_error_detectable": max_denominator_trap,
            "threshold_scope": "q >= 1-a is uniform over L for the specified policy; subthreshold failure needs a sufficiently large L."}


def check_bounds(c, m):
    cases = 0
    for a, b in AB:
        for L in (1.25, 2., 4., 10., 25.):
            theta0 = min(-1., math.log(a * (1 - b) / (4 * b * L)) / (L - 1) - .5)
            for theta in (theta0, theta0 - 1, theta0 - 3):
                bound = -a * (1 - b) / 4 * math.exp(theta) + b * L * math.exp(L * theta)
                c.require(bound < 0, "Eq24 bound strictly negative")
                for q in (0., (1 - a) / 2, 1 - a, 1.):
                    n = audited_field(theta, a, b, L, q)["numerator"]
                    tol = 2e-12 * max(abs(n), abs(bound), 1e-300)
                    c.require(n <= bound + tol, "Eq24 explicit negative-tail bound")
                    c.require(n < 0, "negative tail numerator")
                    cases += 1
            for theta in (-3., -.1, 0., .1, 3.):
                for rule in ("R", "S"):
                    for q in ((0.,) if rule == "R" else (0., .3, 1.)):
                        e = enumeration(theta, a, b, L, rule, q)
                        c.require(abs(e["gradient"]) <= L * (1 + 1e-12), "LemmaA.1 |gradient| <= L")
    return {"negative_tail_cases": cases, "bound": "N_S,q <= -a(1-b)exp(theta)/4 + bL exp(L theta)"}


def check_budget(c, m):
    series = []
    for a in (.4, .75, .9):
        fractions = []
        for n in (1, 2, 4, 16, 64, 256, 1024):
            b = a / (n + 1)
            q = 1 - a
            L = 2 * a * (1 - b) / (b * (1 - a))
            e = enumeration(0, a, b, L, "S", q)
            B = q * b * (a + 1) / (a + b)
            c.close(e["query_fraction"], B, "vanishing-budget exact count")
            c.require(enumeration(0, a, b, L)["gradient"] > 0, "family has unaudited initial harm")
            c.require(e["gradient"] < 0, "family has audited initial improvement")
            c.require(B <= (1 - a) * (a + 1) / a * b + 1e-14, "linear query-fraction upper bound")
            fractions.append(B)
            series.append(dict(a=a,n=n,b=b,L=L,q=q,B=B))
        c.require(all(x > y for x,y in zip(fractions, fractions[1:])), "query fractions decrease")
    return {"series": series, "scope": "Initial expected fraction; finite sequence checks do not prove a limit or uniform convergence time."}


def check_finite_pool(c, m):
    common = np.array([[1.,2.,-1.],[-3.,1.,2.],[2.,-2.,1.]])
    er = np.array([[3.,-1.,2.],[-1.,2.,4.]])
    es = np.array([[3.,-1.,2.],[2.,-3.,1.]]) # First error pair deliberately coincides.
    K = len(common) + len(er)
    R, S = np.vstack([common, er]), np.vstack([common, es])
    gr, gs = R.mean(0), S.mean(0)
    differences = es - er
    rhs = differences.sum(0) / K
    c.close(gs-gr, rhs, "Prop4.7 correct-response cancellation")
    c.close((R/K).sum(0), gr, "recorder contributions already sum to G_R")
    c.close((S/K).sum(0), gs, "recorder contributions already sum to G_S")
    c.require(np.linalg.norm((S/K).sum(0)/K-gs) > .01, "fixture detects double division by K")
    d = np.linalg.norm(differences,axis=1)
    c.require(np.linalg.norm(gs-gr) <= d.sum()/K + 1e-14, "finite-pool norm bound")
    h = np.array([2.,-.5,3.])
    c.require(abs(h@(gs-gr)) <= np.linalg.norm(h)*d.sum()/K + 1e-14, "projection bound")
    c.close(d[0],0,"coincident error retained in denominator")
    return {"K":K,"correct_count":len(common),"error_count":len(er),"gradient_difference":rhs.tolist(),
            "scope":"Deterministic vector fixture, not measured LoRA gradients or decomposed AdamW updates."}


def check_smooth_loss(c, m):
    cases = 0
    Q = np.array([[2.,.3,0],[.3,-1.,.2],[0,.2,.5]]) # Indefinite: no convexity assumption.
    h = np.array([1.2,-.7,.4])
    M = float(np.max(np.abs(np.linalg.eigvalsh(Q))))
    loss = lambda d: float(h@d + .5*d@Q@d)
    pairs = (([.02,-.03,.01],[-.05,.02,.04]),([.3,.2,-.1],[-.2,.1,.1]),([0,0,0],[.1,0,0]))
    for rv,sv in pairs:
        dr,ds=np.array(rv),np.array(sv)
        for d in (dr,ds):
            rem=loss(d)-h@d
            c.close(rem,.5*d@Q@d,"quadratic exact Taylor remainder")
            c.require(abs(rem)<=M*np.dot(d,d)/2+1e-14,"Prop4.8 single-branch bound")
            projection=float(h@d);radius=M*np.dot(d,d)/2
            if projection>radius: c.require(loss(d)>0,"strict positive margin")
            if projection<-radius: c.require(loss(d)<0,"strict negative margin")
            cases+=1
        diff=loss(ds)-loss(dr);projection=float(h@(ds-dr))
        radius=M*(ds@ds+dr@dr)/2
        c.require(abs(diff-projection)<=radius+1e-14,"Prop4.8 paired bound")
        if projection>radius: c.require(diff>0,"paired positive margin")
        if projection<-radius: c.require(diff<0,"paired negative margin")
    # At equality the loss change can be exactly zero, for either projection sign.
    for curvature,linear in ((-2.,1.),(2.,-1.)):
        delta=1.; projection=linear*delta; radius=abs(curvature)*delta**2/2
        c.close(abs(projection),radius,"margin equality fixture")
        c.close(linear*delta+.5*curvature*delta**2,0,"equality does not give a strict sign")
    # Inside the margin interval, even the same positive projection permits either sign.
    c.require(.1+.5*2>0 and .1-.5*2<0,"insufficient margin leaves sign undetermined")
    c.require(np.linalg.norm(h/h.dot(h)**.5-h)>.01,"fixture distinguishes normalized h")
    # Nonquadratic logistic loss with a global Hessian bound, independent quadrature.
    X=np.array([[1.,2.],[-2.,1.],[.5,-1.],[3.,.2]])
    y=np.array([1.,0.,1.,0.]);theta=np.array([.2,-.1]);delta=np.array([.3,-.4])
    n=len(y); H=lambda v: X.T@(1/(1+np.exp(-(X@v)))-y)/n
    L=lambda v: float(np.mean(np.logaddexp(0,X@v)-y*(X@v)))
    hlog=H(theta); Mlog=float(np.linalg.eigvalsh(X.T@X/(4*n))[-1])
    actual=L(theta+delta)-L(theta); rem=actual-hlog@delta
    u,w=np.polynomial.legendre.leggauss(24); u=(u+1)/2;w=w/2
    integral=sum(float((H(theta+t*delta)-hlog)@delta)*weight for t,weight in zip(u,w))
    c.close(rem,integral,"nonquadratic integral remainder",atol=2e-14)
    c.require(abs(rem)<=Mlog*(delta@delta)/2+1e-14,"nonquadratic known-Hessian bound")
    return {"quadratic_cases":cases,"known_quadratic_M":M,"known_logistic_M":Mlog,
            "scope":"M is justified for these fixtures only, not estimated or certified for an LLM; reference NLL is not task risk."}


def check_flow(c, m):
    trajectories={}
    for rule in ("R","S"):
        paths=[]
        for dt in (.01,.005,.0025):
            rows=m.population_flow(0,rule,dt,round(20/dt))
            theta=np.array([r['theta'] for r in rows]);acc=np.array([r['sampling_accuracy'] for r in rows])
            c.require(len(rows)==round(20/dt)+1,"flow sample count")
            c.require(np.all(np.isfinite(theta)) and np.all(np.isfinite(acc)),"finite flow")
            c.require(np.all(np.diff(acc)>=-2e-13) if rule=='R' else np.all(np.diff(acc)<=2e-13),"flow direction")
            # Sample the complete trajectory at fixed intervals against the oracle.
            for idx in np.linspace(0,len(rows)-1,17,dtype=int):
                e=enumeration(theta[idx],.75,.5,10,rule)
                c.close(acc[idx],e['accuracy'],"flow accuracy oracle")
                c.close(rows[idx]['gradient'],e['gradient'],"flow gradient oracle")
            paths.append(acc)
        e1=float(np.max(np.abs(paths[0]-paths[1][::2])))
        e2=float(np.max(np.abs(paths[1]-paths[2][::2])))
        c.require(e2<e1 and e2<.001,"Euler refinement: difference decreases")
        c.require(1.8<e1/e2<2.2,"observed first-order refinement")
        trajectories[rule]={'accuracy_t20':float(paths[-1][-1]),'coarse_difference':e1,'fine_difference':e2}
    # Independent bisection on enumerated outcomes; not a uniqueness proof.
    lo,hi=-.112,0.
    c.require(enumeration(lo,.75,.5,10)['gradient']<0<enumeration(hi,.75,.5,10)['gradient'],"root bracket")
    for _ in range(70):
        mid=(lo+hi)/2
        if enumeration(mid,.75,.5,10)['gradient']<0:lo=mid
        else:hi=mid
    root=(lo+hi)/2
    c.close(m.find_balance('S',-.112,0),root,"root agrees with independent enumeration")
    c.close(root,-.11146044018345,"paper stationary-point readout")
    c.require(trajectories['R']['accuracy_t20']<1,"finite-time readout not asymptotic limit")
    return {'trajectories':trajectories,'S_root':root,'scope':'Finite numerical trajectories; refinement differences are not rigorous exact-solution error bounds.'}


GROUPS = (
    ('general_logistic','Lemma 4.1; Eqs. (5)-(8)',check_general),
    ('initial_and_table4','Proposition 4.2; Corollary 4.4; Table 4',check_initial),
    ('conditional_covariance','Lemma 3.1',check_covariance),
    ('targeted_auditing','Theorem 4.5; Eq. (25)-(26); contrast equality',check_auditing),
    ('tail_and_magnitude_bounds','Lemma A.1; Eq. (24)',check_bounds),
    ('query_budget','Corollary 4.6',check_budget),
    ('finite_pool','Proposition 4.7; Eq. (16)',check_finite_pool),
    ('smooth_loss','Proposition 4.8; Corollary 4.9',check_smooth_loss),
    ('population_flow','Theorem 4.3 numerical illustration; Figure 1',check_flow),
)


def load_logistic(code_dir):
    path=Path(code_dir).resolve()/'rsi/logistic.py'
    if not path.is_file():raise FileNotFoundError(f'Missing {path}')
    spec=importlib.util.spec_from_file_location('rsi_theory_target',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,path


def run_checks(module):
    results=[]
    for name,paper,fn in GROUPS:
        c=Checks()
        try:
            details=fn(c,module)
            results.append(dict(name=name,paper=paper,status='passed',assertions=c.count,max_abs_error=c.max_abs_error,details=details))
        except Exception as exc:
            results.append(dict(name=name,paper=paper,status='failed',assertions=c.count,error=f'{type(exc).__name__}: {exc}'))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,help='Optional new JSON output file; existing files are never overwritten.')
    args=p.parse_args()
    if args.out and args.out.exists():p.error(f'Output already exists: {args.out}')
    try:module,source=load_logistic(args.code_dir)
    except (OSError,ImportError) as exc:p.error(str(exc))
    groups=run_checks(module)
    passed=all(g['status']=='passed' for g in groups)
    report=dict(status='passed' if passed else 'failed',created_at=datetime.now(timezone.utc).isoformat(),
                manuscript_sha256=PAPER_SHA256,source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=platform.python_version(),numpy=np.__version__,
                optimized_mode=bool(sys.flags.optimize),groups=groups,
                scope='Finite checks, not universal proofs, GPU results, or LLM smoothness certification.')
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'groups_passed':sum(g['status']=='passed' for g in groups),
                      'groups_total':len(groups),'assertions':sum(g['assertions'] for g in groups),
                      'failures':[g for g in groups if g['status']=='failed'],'out':str(args.out) if args.out else None},ensure_ascii=False,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
