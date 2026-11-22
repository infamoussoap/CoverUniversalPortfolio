"""Binary64 evaluation of the pair-series lower endpoint for screening."""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def raising(base, h):
    out = base.copy()
    for i in range(1, len(out)):
        out[i] += h*out[i-1]
    return out


@njit(cache=True)
def moments(score):
    d = len(score)
    shift = score.min()
    h = score-shift
    degree = max(30, 4*int(h.max()+1)+16)
    base = np.zeros(degree+1)
    base[0] = 1.
    for u in h:
        base = raising(base, u)
    den = np.ones(degree+1)
    for j in range(1, len(den)):
        den[j] = den[j-1]/(d+j-1)
    # Lemma A.1 uses the shorter mass prefix; A.3 raises the longer one.
    size = min(len(den), max(24, int(4*h.max())+1)+1)
    mgf = math.exp(shift)*(base[:size] @ den[:size])
    return h, base, mgf


@njit(cache=True)
def pair_contractions(g, h, base, Q, K, C):
    d = len(h)
    den = np.ones(len(base))
    for i in range(1, len(den)):
        den[i] = den[i-1]/(d+i+1)
    single = np.empty(d)
    raised = np.empty((d,len(base)))
    for i in range(d):
        raised[i] = raising(base,h[i])
        single[i] = raised[i] @ den
    q, k, c = 0., 0., 0.
    for i in range(d):
        for j in range(i+1):
            diff = h[i]-h[j]
            if abs(diff) > (1+abs(h[i])+abs(h[j]))/1024:
                f = (h[i]*single[i]-h[j]*single[j])/diff
            else:
                f = raising(raised[i],h[j]) @ den
            # Diagonal prior factor two; off-diagonal occurs twice.
            q += 2*f*Q[i,j]
            k += 2*f*K[i,j]
            c += 2*f*C[i,j]
    scale = math.exp(g.min())/(d*(d+1))
    return q*scale, k*scale, c*scale


@njit(cache=True)
def quad_terms(A, p, h, rising):
    d = len(A)
    v = np.zeros(5)
    for n in range(5):
        for i in range(d):
            v[n] += (n+1)*A[i,i]*p[n,i]
        for j in range(1,n):
            for i in range(d):
                for k in range(d):
                    v[n] += p[j,i]*A[i,k]*p[n-j,k]
    q = np.zeros(5)
    for n in range(5):
        for j in range(n+1):
            q[n] += h[n-j]*v[j]
        q[n] /= rising[n+2]
    return q


@njit(cache=True)
def exp_tail(x, degree):
    fac = 1.
    for k in range(2,degree+1):
        fac *= k
    if x <= .001:
        return math.exp(x)/fac
    # Positive Taylor summation avoids near-zero cancellation.
    term = 1./fac
    total = term
    for k in range(degree+1,degree+201):
        term *= x/k
        total += term
        if term < 1e-17*total:
            break
    return total


@njit(cache=True)
def lower(g, h, base, mgf, Q, K, C, J, diag, weighted, r4, rising):
    d = len(g)
    p = np.ones((5,d))
    for k in range(1,5):
        p[k] = g*p[k-1]
    hpoly = np.array([1.,0.,p[2].sum()/2,p[3].sum()/3,
                      p[2].sum()**2/8+p[4].sum()/4])
    quad = quad_terms(Q,p,hpoly,rising)
    ab = quad_terms(C,p,hpoly,rising)
    peak = g.max()
    qhigh = quad[:4].sum()+exp_tail(peak,4)*24*quad[4]
    err = exp_tail(peak,3)*math.sqrt(48*max(ab[2],0.)*max(ab[4],0.))
    ej = 2*diag.sum()/rising[3]
    egj = (3*(g @ weighted)+6*(g @ diag))/rising[4]
    eg2j = (2*p[2].sum()*diag.sum()+12*(g @ J @ g)
            +6*(p[2] @ weighted)+24*(p[2] @ diag))/rising[5]
    q, k, c = pair_contractions(g, h, base, Q, K, C)
    j_low = max(ej + egj + eg2j / 2 - err, -c)
    ah = min(q, qhigh) - j_low / 3 + min(k, math.exp(peak) * r4)
    return math.log(mgf)-max(ah,0.)/mgf


@njit(cache=True)
def screen(x, full=False):
    T, d = x.shape
    out = np.full(T,-np.inf)
    Q, K, C, J = np.zeros((4,d,d))
    g, diag, weighted = np.zeros((3,d))
    r4 = 0.
    rising = np.ones(9)
    for i in range(1,9):
        rising[i] = rising[i-1]*(d+i-1)
    limit = math.log(20.)-.05
    for t in range(T):
        a = x[t]/x[t].mean()-1
        g += a
        low = 1+a.min()
        span = max(abs(a.min()),abs(a.max()))
        v = a*a
        norm = v.sum()
        diag += a*v
        weighted += a*norm
        r4 += (3*norm*norm+6*(v @ v))/(4*low*rising[4])
        for i in range(d):
            for j in range(d):
                u = a[i]*a[j]
                Q[i,j] += u/2
                K[i,j] += u*(span*span/(4*low))
                C[i,j] += u*span
                J[i,j] += a[i]*u
        if not full and g.max() < limit-1e-8:
            continue
        if not full and np.exp(g).mean() < math.exp(limit)-1e-8:
            continue
        h, base, mgf = moments(g)
        # Bound the omitted positive mass terms by the geometric tail.
        rho = h.max()
        n = max(24,int(4*rho)+1)
        if not full and n+1 < len(base):
            ratio = rho/(n+1)
            weighted_term = base[n]
            for j in range(n):
                weighted_term /= d+j
            tail = math.exp(g.min())*weighted_term*ratio/(1-ratio)
            if mgf+tail < math.exp(limit)-1e-8:
                continue
        out[t] = lower(g,h,base,mgf,Q,K,C,J,diag,weighted,r4,rising)
    return out
