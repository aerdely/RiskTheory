### Exercise: Collective Risk Model
### Author: Dr. Arturo Erdely
### Version: 2024-09-02

#=
    Let X be a Pareto random variable with parameters β > 0 and θ > 0
    with pdf:    f(x|β,θ) = β⋅(θ^β) / (x^(β+1))  ,  x ≥ θ.

    Recall that Median(X) = θ⋅(2^(1/β))

    Consider a portfolio of 1-year term property and casualty insurance,
    under the collective risk model:    S = Y1 + ⋯ + YN 
    where the frequency is a random variable: N = max{n ∈ {0,1,…}: n ≤ X-θ}
    and the conditional severity per claim is: Y|N=n ~ Pareto(2+1/n, δ) for n ≥ 1.

    Notice that Median(Y|N=n) = δ⋅(2^(1/(2+1/n))) ↑ if n↑

    Calculate or estimate the expected value, variance, median, and 99.5% VaR
    of S and S|S>0, with parameter values β = 3, θ = 1 = δ.
=#


## Load needed packages and code

begin
    using Plots, Distributions
    include("02probestim.jl")
end

@doc Pareto



## Collective risk model simulator

function simulateCRM(; β = 3.0, θ = 1.0, δ = 1.0, m = 1_000_000)
    S = zeros(m)
    X = Pareto(β, θ)
    N = Int.(floor.(rand(X, m) .- θ)) # m = number of simulations
    iN = findall(N .≥ 1)  # positions in N such that N ≥ 1
    Y = []
    for i ∈ iN
        Yi = rand(Pareto(2 + 1/N[i], δ), N[i]) # Y|N=n ~ Pareto(2+1/n, δ)
        S[i] = sum(Yi)
        append!(Y, Yi)
    end
    PS0 = 1 - count(S .> 0)/m # <-- P(S = 0)
    return (ES = mean(S), MS = median(S), VS = var(S), PS0 = PS0,
            VaRS = quantile(S, 0.995), S = S, N = N, Y = Y
    )
end


## Calculations about S

begin # run several times to check differences
    @time sim = simulateCRM(m = 10_000_000)
    println("E(S) = ", sim.ES)
    println("M(S) = ", sim.MS)
    println("V(S) = ", sim.VS)
    println("VaR(S) = ", sim.VaRS)
    println("min(S), max(S) = ", extrema(sim.S))
    println("P(S = 0) = ", sim.PS0)
end

length(sim.S), length(sim.N)
count(sim.N .> 0)
sum(sim.N), length(sim.Y)

## Calculations about S|S>0  (positive total claims)

claims = sim.S[findall(sim.S .> 0)]
begin
    println("E(S|S>0) = ", mean(claims))
    println("M(S|S>0) = ", median(claims))
    println("V(S|S>0) = ", var(claims))
    println("VaR(S|S>0) = ", quantile(claims, 0.995))
    println("min(S|S>0), max(S|S>0) = ", extrema(claims))
end

allunique(claims) # S|S>0 continuous random variable model

c = densprob(claims);
keys(c)
ss = collect(range(c.min, c.max, length = 1_000))
plot(ss, c.fdp.(ss), lw = 2, legend = false, xlabel = "S | S > 0", ylabel = "density")

logc = densprob(log.(claims));
logss = collect(range(logc.min, logc.max, length = 1_000))
plot(logss, logc.fdp.(logss), lw = 2, legend = false, xlabel = "log(S) | S > 0", ylabel = "density")
