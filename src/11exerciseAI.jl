################################################################################
# Risk Theory I — Homework 2027-1 — Exercise 1.7
#
# Claude AI model Opus 5 - Sept 17/2026
#
# Prompt: Solve exercise 1.7 from the attached homework.pdf file. Also attached
#         the required data file CRMdata.txt. Use Julia programming wherever
#         needed. Add detailed explanations as comments in a downloadable file
#         with Julia code.
#
# Answer:
#
# Collective risk model for weekly total claims S = Y_1 + ... + Y_N
# (amounts in million pesos), fitted to CRMdata.txt.
#
# MODEL (allows dependence between frequency and severity):
#   Frequency :  N ~ discrete Pareto(b, t),  P(N ≥ k) = (1 + k/t)^(-b), k = 0,1,...
#   Severity  :  Y_1,...,Y_n | N = n  iid ~ Pareto(α_n, θ = 5),  α_n = A + B/n
#
# The classical collective model assumes N independent of the Y's. The data
# reject that assumption (LRT p ≈ 2e-5): weeks with many claims have heavier
# severity tails (smaller α_n). The case B = 0 recovers the independent model.
#
# Required packages:  ] add Distributions Optim
################################################################################

using Distributions   # Chisq distribution for the likelihood ratio test
using Optim           # Nelder–Mead numerical maximization of log-likelihoods
using Random          # Xoshiro reproducible random number generator
using Statistics      # mean, var, median, quantile


################################################################################
## 1) DATA
################################################################################
# Each row of CRMdata.txt is ONE week in which at least one claim was filed;
# the comma-separated values in the row are the individual claim amounts.
# Weeks without claims are NOT in the file, so they must be added back.

rows = [parse.(Float64, split(line, ',')) for line in readlines("CRMdata.txt")
        if !isempty(strip(line))]          # skip a possible empty final line

n = length.(rows)            # number of claims in each week with claims (≥ 1)
y = reduce(vcat, rows)       # all individual claim amounts in one vector

# For every claim amount, store the number of claims N in ITS week.
# This is needed because the severity distribution depends on N.
ny = reduce(vcat, [fill(length(r), length(r)) for r in rows])

# ASSUMPTION: 20 years ≈ 20 × 365.25 / 7 ≈ 1043.6 → W = 1044 weeks.
# (Using W = 1040 changes E[S] and VaR by less than 1%.)
W = 1044

# Full frequency sample: observed counts plus W − 262 weeks with N = 0.
N = vcat(n, zeros(Int, W - length(n)))

# Pareto lower bound. The smallest amount is 5.003, which suggests a
# reporting threshold / deductible of 5; θ is fixed there (not estimated).
θ = 5.0

# Empirical weekly totals (zero for weeks without claims), for comparison.
S_emp = vcat(sum.(rows), zeros(W - length(n)))


################################################################################
## 2) MAXIMUM LIKELIHOOD ESTIMATION
################################################################################
# The joint likelihood of the data factorizes as
#     L(b, t, A, B) = ∏_weeks P(N = n_i | b, t) × ∏_claims f(y_ij | n_i, A, B),
# so the MLE of (b, t) and of (A, B) can be obtained separately.

# --- 2a) Frequency: discrete Pareto -----------------------------------------
# Survival function P(N ≥ k). This is the law of N = ⌊X⌋ with X ~ Lomax(b, t),
# i.e. P(X > x) = (1 + x/t)^(-b); same structure as N = ⌊X − θ⌋ in Exercise 1.6.
survN(k, b, t) = (1 + k / t)^(-b)

# Probability mass function: P(N = k) = P(N ≥ k) − P(N ≥ k + 1).
pmfN(k, b, t) = survN(k, b, t) - survN(k + 1, b, t)

"""
    fitN(N; x0)

MLE of (b, t) for the discrete Pareto from a vector of counts `N`.
Optimization is done on (log b, log t) so both parameters stay positive
without constraints.
"""
function fitN(N; x0 = [0.7, 0.0])
    function negloglik(p)
        bb, tt = exp(p[1]), exp(p[2])
        return -sum(log(pmfN(k, bb, tt)) for k in N)
    end
    opt = optimize(negloglik, x0, NelderMead())
    return exp.(Optim.minimizer(opt))      # back-transform to (b, t)
end

# --- 2b) Severity: Pareto(α, θ) with α depending on N ------------------------
# Pareto pdf: f(y) = α θ^α / y^(α+1),  y ≥ θ. Log-likelihood summed over claims;
# `α` can be a scalar (constant α) or a vector (one α_n per claim).
loglikY(α, y) = sum(log.(α) .+ α .* log(θ) .- (α .+ 1) .* log.(y))

"""
    fitY(y, ny; x0)

MLE of (A, B) in α_n = A + B/n, where `ny[j]` is the number of claims in the
week of claim `y[j]`. Parameter values giving some α_n ≤ 0 are rejected.
"""
function fitY(y, ny; x0 = [2.0, 1.0])
    function negloglik(p)
        α = p[1] .+ p[2] ./ ny
        any(α .<= 0) && return Inf
        return -loglikY(α, y)
    end
    return Optim.minimizer(optimize(negloglik, x0, NelderMead()))
end

b, t = fitN(N)       # frequency parameters  (≈ 2.05, 1.03)
A, B = fitY(y, ny)   # severity parameters   (≈ 1.77, 1.06)

# --- 2c) Test of independence: H0: B = 0 (constant α) vs H1: B ≠ 0 ----------
# Under H0 the MLE is closed form: α̂0 = m / Σ log(y_j / θ).
α0  = length(y) / sum(log.(y ./ θ))
LRT = 2 * (loglikY(A .+ B ./ ny, y) - loglikY(α0, y))
pvalue = ccdf(Chisq(1), LRT)   # Wilks: LRT ≈ χ²(1) under H0 (1 extra parameter)

println("Frequency : b = $b, t = $t")
println("Severity  : A = $A, B = $B   (constant-α MLE under independence: $α0)")
println("Independence test: LRT = $LRT, p-value = $pvalue")


################################################################################
## 3) EXACT EXPECTED VALUE OF S
################################################################################
# By the tower property, conditioning on N:
#     E[S] = Σ_{n≥1} P(N = n) · n · E[Y | N = n],
#     E[Y | N = n] = θ α_n / (α_n − 1)   (finite only if α_n > 1).
# The series is truncated at K; since P(N = n) ~ n^(-b-1), the omitted tail
# is of order K^(1-b), negligible for K = 2 million.
#
# NOTE ON THE VARIANCE: Var(Y | N = n) = θ² α_n / ((α_n − 1)² (α_n − 2)) is
# infinite whenever α_n ≤ 2. With Â = 1.77 this holds for all n ≥ 5, hence
# Var(S) = ∞ under the fitted model, and no variance is computed.
function ES(b, t, A, B; K = 2_000_000)
    s = 0.0
    for k in 1:K
        α = A + B / k
        α <= 1 && return Inf          # E[Y | N = k] does not exist
        s += pmfN(k, b, t) * k * θ * α / (α - 1)
    end
    return s
end


################################################################################
## 4) MONTE CARLO SIMULATION OF S
################################################################################
# Median and VaR have no closed form, so S is simulated:
#   • N by inversion of the Lomax cdf and flooring:
#         X = t((1 − U)^(-1/b) − 1),  N = ⌊X⌋.
#     (1 − U) ∈ (0, 1] avoids raising 0 to a negative power.
#   • Given N = k, draw k Pareto(α_k, θ) amounts by inversion:
#         Y = θ (1 − U)^(-1/α_k).
#   • S = sum of the k amounts (S = 0 when k = 0).
# The max(·, 1e-3) guard only matters inside the bootstrap, where a resampled
# (A, B) could produce α_k ≤ 0 for large k.
function simS(M, b, t, A, B; rng = Xoshiro(2027))
    S = zeros(M)
    for i in 1:M
        k = floor(Int, t * ((1 - rand(rng))^(-1 / b) - 1))
        k == 0 && continue                 # week without claims: S = 0
        α = max(A + B / k, 1e-3)           # tail index for this week's claims
        s = 0.0
        for _ in 1:k
            s += θ * (1 - rand(rng))^(-1 / α)
        end
        S[i] = s
    end
    return S
end

S   = simS(10^7, b, t, A, B)   # 10 million simulated weeks
Spos = S[S .> 0]               # conditional sample S | S > 0

# Unconditional results.
# The median is 0 because P(S = 0) = P(N = 0) = 1 − (1 + 1/t)^(-b) ≈ 0.75 > 0.5.
println("\nS:        E[S] = ", ES(b, t, A, B), "  (simulated ", mean(S), ")",
        "   median = ", median(S), "   VaR 99.5% = ", quantile(S, 0.995))

# Conditional on S > 0:  E[S | S > 0] = E[S] / P(S > 0), with P(S > 0) = P(N ≥ 1).
println("S | S>0:  E = ", ES(b, t, A, B) / survN(1, b, t),
        "   median = ", median(Spos), "   VaR 99.5% = ", quantile(Spos, 0.995))

# Empirical counterparts for a sanity check of the fitted model.
println("Empirical: E = ", mean(S_emp), "   median = ", median(S_emp),
        "   VaR 99.5% = ", quantile(S_emp, 0.995))


################################################################################
## 5) PARAMETRIC BOOTSTRAP: 95% CONFIDENCE INTERVALS
################################################################################
# Point estimates ignore parameter uncertainty, which is large here because
# both tails (b ≈ 2, A < 2) are heavy. For each replicate r = 1, ..., R:
#   1. Simulate a new data set of W weeks from the FITTED model (b, t, A, B):
#      counts N_r and, for each claim, an amount from Pareto(A + B/n, θ).
#   2. Re-estimate (b_r, t_r, A_r, B_r) by maximum likelihood on that data set.
#   3. Recompute E[S] (exact series) and VaR / medians (simulation with M weeks).
# The 2.5% and 97.5% percentiles across replicates give 95% percentile CIs.
# Runtime: several minutes for R = 1000, M = 500_000.
function bootstrap(R; M = 500_000, rng = Xoshiro(7))
    out = zeros(R, 4)
    for r in 1:R
        # Step 1: synthetic data set of the same size as the real one
        Nr  = [floor(Int, t * ((1 - rand(rng))^(-1 / b) - 1)) for _ in 1:W]
        nyr = reduce(vcat, [fill(k, k) for k in Nr if k > 0])
        yr  = [θ * (1 - rand(rng))^(-1 / (A + B / k)) for k in nyr]
        # Step 2: refit the model
        br, tr = fitN(Nr)
        Ar, Br = fitY(yr, nyr)
        # Step 3: risk measures under the refitted model
        Sr = simS(M, br, tr, Ar, Br; rng = rng)
        out[r, :] = [ES(br, tr, Ar, Br),
                     quantile(Sr, 0.995),
                     median(Sr[Sr .> 0]),
                     quantile(Sr[Sr .> 0], 0.995)]
    end
    return out
end

boot = bootstrap(1000)
println("\nParametric bootstrap, percentiles [2.5%, 50%, 97.5%]:")
for (j, name) in enumerate(["E[S]", "VaR99.5(S)", "Median(S|S>0)", "VaR99.5(S|S>0)"])
    println(rpad(name, 16), quantile(boot[:, j], [0.025, 0.5, 0.975]))
end
