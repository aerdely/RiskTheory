## Student distribution in Julia
## Author: Arturo Erdely
## Version date: 2024-02-11

## Functions: rvStudent, rvStudent3

using Distributions


"""
    rvStudent(ν::Real)

**Student** standard probability distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)
with parameter `ν > 0` degrees of freedom (df). 

For example, if we define `X = rvStudent(3.5)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode
- `X.mean` = theoretical mean
- `X.var` = theoretical variance

Dependencies: 
> `Distributions` (external) package

## Example
```
X = rvStudent(3.5);
keys(X)
println(X.model)
X.param
X.param.df
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range
X.mean, X.var
xsim = X.sim(10_000); # a random sample of size 10,000
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
using Statistics # Julia standard package
median(xsim), mean(xsim), var(xsim) # sample median, mean, and variance
```
"""
function rvStudent(ν::Real)
    if ν ≤ 0
        error("Parameter must be positive")
        return nothing
    else
        S = TDist(ν)
        dst(x) = pdf(S, x)
        pst(x) = cdf(S, x)
        qst(u) = 0 < u ≤ 1 ? quantile(S, u) : NaN
        rst(n::Integer) = qst.(rand(n))
    end
    soporte = ("]-∞ , ∞[", -Inf, Inf)
    mediana = 0.0
    ric = qst(0.75) - qst(0.25)
    moda = 0.0
    media = ν > 1 ? 0.0 : NaN
    if ν > 2
        varianza = ν / (ν-2)
    elseif 1 < ν ≤ 2
        varianza = Inf
    else
        varianza = NaN 
    end
    return (model = "Student", param = (df = ν,), range = soporte,
            pdf = dst, cdf = pst, qtl = qst, sim = rst,
            median = mediana, iqr = ric, mode = moda, mean = media, var = varianza
    )
end


"""
    rvStudent3(μ::Real, σ::Real, ν::Real)

Location-dispersion **Student** probability distribution with location
parameter `μ`, dispersion parameter `σ > 0`, and `ν > 0` degrees of freedom (df). 

For example, if we define `X = rvStudent3(-2.0, 1.7, 3.5)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode
- `X.mean` = theoretical mean
- `X.var` = theoretical variance

Dependencies: 
> `rvStudent` function

## Example
```
X = rvStudent3(-2.0, 1.7, 3.5);
keys(X)
println(X.model)
X.param
X.param.dis
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range
X.mean, X.var
xsim = X.sim(10_000); # a random sample of size 10,000
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
using Statistics # Julia standard package
median(xsim), mean(xsim), var(xsim) # sample median, mean, and variance
```
"""
function rvStudent3(μ::Real, σ::Real, ν::Real)
    if min(σ, ν) ≤ 0
        error("Parameters σ and ν must be positive")
        return nothing
    else
        S = rvStudent(ν)
        dst3(x) = S.pdf((x - μ) / σ) / σ
        pst3(x) = S.cdf((x - μ) / σ)
        qst3(u) = 0 < u ≤ 1 ? μ +σ*S.qtl(u) : NaN
        rst3(n::Integer) = qst3.(rand(n))
    end
    soporte = ("]-∞ , ∞[", -Inf, Inf)
    mediana = μ
    ric = qst3(0.75) - qst3(0.25)
    moda = μ
    media = ν > 1 ? μ : NaN
    if ν > 2
        varianza = (σ^2)*ν / (ν-2)
    elseif 1 < ν ≤ 2
        varianza = Inf
    else
        varianza = NaN 
    end
    return (model = "Student3", param = (loc = μ, dis = σ, df = ν), range = soporte,
            pdf = dst3, cdf = pst3, qtl = qst3, sim = rst3,
            median = mediana, iqr = ric, mode = moda, mean = media, var = varianza
    )
end

@info "rvStudent  rvStudent3"
