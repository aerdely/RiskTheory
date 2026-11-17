### Exercise: Ruin Theory
### Author: Dr. Arturo Erdely
### Version date: 2026-11-10


## Packages
begin
    using Random, Statistics # Julia standard library
    using Distributions, StatsBase, StatsPlots, Optim # external packages
end



## Collective risk model

# S = Y(1) + ⋯ + Y(N) 
# where the frequency is a random variable N ~ BinomialNegativa(r, θ) with N ∈ {0, 1,...}
# and the random variables {Y(1), Y(2), ...} represent claim severities, but conditionally
# independendient given {N = n} with conditional distribution:
# Y | N = n ~ LogNormal(μ(n), σ(n)) where:
# μ(n) = g1(n) + ε1  with ε1 ~ Normal(0, ω1)
# σ(n) = g2(n) + ε2  with ε2 ~ Normal(0, ω2)



## Read and process data

begin
    contenido = readlines("siniestros.csv")    
    m = length(contenido)
    sev = fill(Float64[], m) # [Y1, Y2, YN]
    frec = zeros(Int, m) # N
    reclam = zeros(m) # S
    for i ∈ 1:m
        v = parse.(Float64, split(contenido[i], ','))
        if v == [0.0]
            sev[i] = []
        else
            sev[i] = v # vectors of severities
            frec[i] = length(v) # frequency
            reclam[i] = sum(v) # total claims
        end
    end
end

function resumen(x)
    (median = median(x), mean = mean(x), quartile3 = quantile(x, 0.75), VaR = quantile(x, 0.995))
end

begin
    println("Severities per period:")
    display(sev[1:5])
    println("    ⋮        ⋮\n")
    println("Frequency per period: ", frec[1:5], " ...")
    println("Total claims per period: ", reclam[1:5], " ...")
    resumen(reclam)
end



## Exploring the data

# total claims per period
begin
    histogram(reclam, legend = false, color = :yellow)
    title!("Total claims per period")
    xaxis!("amount")
         yaxis!("counts")
    savefig("z01claims.png")
end

# frequency
begin
    rangofrec = collect(minimum(frec):maximum(frec))
    bar(rangofrec, counts(frec), legend = false)
    title!("Claims frequency")
         xaxis!("claims per period")
         yaxis!("counts")
    savefig("z02frequency.png")
end

# severity vs frequency
begin
    frecvalores = setdiff(sort(unique(frec)), [0])
    plot(legend = false, title = "Severity of claims")
    xaxis!("number of claims per period")
    yaxis!("claim amount")
    for k ∈ frecvalores
        ifrec = findall(frec .== k)
        for ℓ ∈ ifrec
            scatter!(fill(k, k), sev[ℓ], color = :gray)
        end
    end
    savefig("z03severity.png")
end

begin
    medias = zeros(length(frecvalores))
    desvst = zeros(length(frecvalores))
    plot(legend = false, title = "Log-severity of claims")
    xaxis!("Number of claims per period")
    yaxis!("Log-severity")
    i = 0
    for k ∈ frecvalores
        global i += 1
        valores = Float64[]
        ifrec = findall(frec .== k)
        for ℓ ∈ ifrec
            append!(valores, log.(sev[ℓ]))
            # scatter!([k], log.(sev[ℓ]), color = :gray)
        end
        scatter!(fill(k, length(valores)), valores, color = :gray)
        medias[i] = mean(valores)
        desvst[i] = std(valores, corrected = false)
    end
    plot!(frecvalores, medias, color = :red, lw = 3)
    plot!(frecvalores, medias .+ 2 .* desvst, color = :blue, lw = 3)
    plot!(frecvalores, medias .- 2 .* desvst, color = :blue, lw = 3)
    savefig("z04logseverity.png")
end

begin
    p1 = plot(frecvalores, medias, color = :red, lw = 3, legend = false)
    title!("Mean log-severity conditional on frequency")
    xaxis!("frequency")
    yaxis!("mean")
    savefig("z05logmeanseverity.png")
    p2 = plot(frecvalores, desvst, color = :blue, lw = 3, legend = false)
    title!("Standard deviation of conditional log-severity")
    xaxis!("frequency")
    p22 = yaxis!("standard deviation")
    savefig("z06logsevstddev.png")
    plot(p1, p2, layout = (2,1), size = (600,600))
end




## Fit model for μ in log-severity ~ Normal(μ(n), σ(n))

# μ(n) = α⋅exp(nβ) + ε1 where ε1 ~ Normal(0, ω1), ω1 = desvest

# initial values for optimization

begin
    g1(n, α, β) = α * exp(n * β)
    α0 = 1.2 * √(2/3)
    β0 = log(3/2) / 20
    n = frecvalores
end;

begin
    plot(frecvalores, medias, color = :red, lw = 3, legend = :left, label = "observed")
    title!("Log-severity")
    xaxis!("frequency")
    yaxis!("conditional mean")
    plot!(n, g1.(n, α0, β0), color = :black, lw = 2, label = "non-optimal smoothing")
end

# optimal parameter values

h(x) = sum((medias .- g1.(n, x[1], x[2])) .^ 2) # sum of quadratic differences
opt_μ = optimize(h, [α0, β0])
Optim.minimizer(opt_μ)
α = Optim.minimizer(opt_μ)[1]
β = Optim.minimizer(opt_μ)[2]

begin
    plot!(n, g1.(n, α, β), color = :green, lw = 2, label = "optimal smoothing")
    savefig("z07suavizamedia.png")
end

begin
    residuos1 = medias .- g1.(n, α, β)
    ω1 = std(residuos1)
    plot(residuos1, legend = false, title = "residuals")
    p3 = scatter!(residuos1)
    p4 = histogram(residuos1, legend= false)
    plot(p3, p4, layout = (1, 2))
end



## Fit model for σ in log-severity ~ Normal(μ(n), σ(n))

p22

# σ(n) = γ⋅(1 - exp(-nδ)) + ε2 where ε2 ~ Normal(0, ω2), ω2 = desvest

# initial values for optimization

begin
    w = range(-1, 1.02, length = 1_000)
    p(w) = 13*(w^30) - 27*w + 14
    plot(w, p.(w), lw = 3, legend = false, xlabel = "w", ylabel = "p(w)")
end

begin
    δ0 = log(2)
    γ0 = 0.26
    g2(n, γ, δ) = γ * (1 - exp(-n * δ))
    plot(frecvalores, desvst, color = :red, lw = 3, legend = :topleft, label = "observed")
    title!("Log-severity")
    xaxis!("frequency")
    yaxis!("standard deviation")
    plot!(n, g2.(n, γ0, δ0), color = :black, lw = 2, label = "non-optimal smoothing")
end

# optimal parameter values

h(x) = sum((desvst .- g2.(n, x[1], x[2])) .^ 2) # sum of quadratic differences
opt_σ = optimize(h, [γ0, δ0])
Optim.minimizer(opt_σ)
γ = Optim.minimizer(opt_σ)[1]
δ = Optim.minimizer(opt_σ)[2]

begin
    plot!(n, g2.(n, γ, δ), color = :green, lw = 2, label = "optimal smoothing")
    savefig("z08suavizadesv.png")
end

begin
    residuos2 = desvst .- g2.(n, γ, δ)
    ω2 = std(residuos2)
    plot(residuos2, legend = false, title = "residuals")
    p5 = scatter!(residuos2)
    p6 = histogram(residuos2, legend= false)
    plot(p5, p6, layout = (1, 2))
end



## Frequency model fit

mean(frec), var(frec)

# Negative Binomial (r, θ) → method of moments

θ = mean(frec) / var(frec)
r = (mean(frec)^2) / (var(frec) - mean(frec))

begin
    bar(rangofrec, proportions(frec), label = "observed", title = "Frequency")
    scatter!(rangofrec, pdf.(NegativeBinomial(r, θ), rangofrec),
             label = "Negative Binomial fit")
    plot!(rangofrec, pdf.(NegativeBinomial(r, θ), rangofrec), label ="")
    savefig("z09frecuencia.png")
end



## Simulate total claims (per week)

function simula(r, θ, α, β, ω1, γ, δ, ω2, numsims = 10_000)
    rS = zeros(numsims)
    rY = fill(Float64[], numsims)
    N = NegativeBinomial(r, θ)
    rN = rand(N, numsims)
    iN = findall(1 .≤ rN)
    for i ∈ iN
        μ = (α * exp(rN[i] * β)) .+ rand(Normal(0, ω1), rN[i])
        σ = γ * (1 - exp(-rN[i] * δ)) .+ rand(Normal(0, ω2), rN[i])
        Y = zeros(rN[i])
        for j ∈ 1:rN[i]
            Y[j] = rand(LogNormal(μ[j], σ[j]), 1)[1]
        end
        rS[i] = sum(Y)
        rY[i] = Y 
    end
    return (reclam = rS, frec = rN, sev = rY)
end


@time sim = simula(r, θ, α, β, ω1, γ, δ, ω2, 1_000); # first time is slower
@time sim = simula(r, θ, α, β, ω1, γ, δ, ω2, 1_000); # second time is faster

@time sim = simula(r, θ, α, β, ω1, γ, δ, ω2, 1_000_000);

resumen(sim.reclam)

resumen(reclam)



## Trying other models for total claims (per week)

begin
    histogram(reclam, normalize = true, label = "observed", color = :yellow)
    title!("Total claims per period")
    xaxis!("amount")
    yaxis!("density")
end

G = fit_mle(Gamma, reclam) # error (because zeros present)
G = fit_mle(Gamma, setdiff(reclam, [0.0])) # ignoring zeros 
LN = fit_mle(LogNormal, setdiff(reclam, [0.0])) # ignoring zeros

begin
    z = range(0.001, 300.0, length = 1_000)
    plot!(z, pdf.(G, z), label = "Gamma", lw = 2)
    plot!(z, pdf.(LN, z), label = "LogNormal", lw = 2)
end

resumen(reclam)
resumen(sim.reclam)
resumen(G)
resumen(LN)



## Simulate total claims (per year)

function simula52(r, θ, α, β, ω1, γ, δ, ω2, numsims = 10_000)
    reclam = zeros(numsims)
    for i ∈ 1:numsims
        sim = simula(r, θ, α, β, ω1, γ, δ, ω2, 52) # simulate 52 weeks = 1 year
        reclam[i] = sum(sim.reclam) # simulated total claims in 1 year
    end
    return(reclam)
end

@time simula52(r, θ, α, β, ω1, γ, δ, ω2, 1_000);

@time simS = simula52(r, θ, α, β, ω1, γ, δ, ω2, 1_000_000); # 35 secs approx



## SCR, initial capital and total premiums

reS = resumen(simS)
ROE = 0.13
iREG = 0.07
SCR = reS.VaR - reS.mean
C0 = (1 - (ROE - iREG)) * SCR
Π = reS.mean + (ROE - iREG)*SCR



## Simulate risk process

function simProceso(r, θ, α, β, ω1, γ, δ, ω2, periodos, numsims = 10_000)
    C = zeros(numsims, periodos + 1)
    C[:, 1] .= C0
    Πacum = collect(1:periodos) .* Π
    for k ∈ 1:numsims
        S = simula52(r, θ, α, β, ω1, γ, δ, ω2, periodos)
        C[k, 2:end] = C0 .+ Πacum .- cumsum(S)
    end
    return C
end


# Expected capital and probability of ruin 

begin
    periodos = 10
    numsims = 100_000
    @time C = simProceso(r, θ, α, β, ω1, γ, δ, ω2, periodos, numsims);
    probRuina = zeros(periodos + 1)
    caProm = zeros(periodos + 1)
    caProm[begin] = C0
    for k ∈ 1:periodos
        probRuina[k+1] = mean(C[:, k+1] .< 0.0)
        caProm[k+1] = mean(C[:, k+1])
    end
end

begin
    display(hcat(collect(0:periodos), caProm, probRuina))
    plot(0:periodos, probRuina, legend = false)
    xaxis!("Year"); title!("Probability of ruin")
end

begin
    np = 1000
    iPath = randperm(numsims)[1:np]
    plot(0:periodos, C[iPath[begin], :], legend = false, xticks = 0:periodos, color = :black)
    title!("Risk process"); xaxis!("Year"); yaxis!("Capital")
    for k ∈ 2:np
        plot!(0:periodos, C[iPath[k], :], color = :black)
    end
    plot!(0:periodos, caProm, color = :green, lw = 3)
    hline!([0.0], color = :red, lw = 2)
    current()
end

# Severity of ruin

begin
    sevRuina = fill(Float64[0.0], periodos + 1)
    sevRuinaProm = zeros(periodos + 1)
    sevRuinaMediana = zeros(periodos + 1)
    for t ∈ 1:periodos
        iRuina = findall(C[:, t+1] .< 0.0)
        if length(iRuina) > 0
            sevRuina[t+1] = -1 .* C[iRuina, t+1]
            sevRuinaProm[t+1] = mean(sevRuina[t+1])
            sevRuinaMediana[t+1] = median(sevRuina[t+1])
        end
    end
end

solution = hcat(collect(0:periodos), caProm, probRuina, sevRuinaProm, sevRuinaMediana)

vcat(["t" "E(C)" "P(C<0)" "E(-C|C<0)" "M(-C|C<0)"], solution)
