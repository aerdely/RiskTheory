### Exercise: Collective Risk Model
### Author: Dr. Arturo Erdely
### Version: 2024-09-01

#=

    Consider a portfolio of 1-year term life insurance independent policies from
    the file `LIFEinsurance.csv` that specifies age and insured amount for each
    policy, with the additional benefit of twice the insured amount in case of
    accidental death, assuming that 1 out of 10 deaths is accidental (regardless
    of the age). Use the mortality table in the file `mortality.csv` 

    Build a collective risk model that approximates the results of the individual
    risk model (see 08exercise.jl).
   	
    Remember that the data can be downloaded from the Github repository of this course:
    https://github.com/aerdely/RiskTheory

=#



## Load needed packages and code

begin
    using CSV, DataFrames, Plots, Distributions
    include("02probestim.jl")
end



## Load data (see 08exercise.jl)

begin
    # insurance portfolio
    policy = CSV.read("LIFEinsurance.csv", DataFrame)
    r = length(policy.AGE) # number of policies 
    # mortality table
    mort = CSV.read("mortality.csv", DataFrame)
    # Dictionary: a function that maps age -> qx
    q = Dict(mort.AGE .=> mort.qx)
    # Add qj to each policy j
    policy.q = zeros(r)
    for j ∈ 1:r
        policy.q[j] = q[policy.AGE[j]]
    end
    k = 1/10 # probability of accidental given death
    display(policy)
end



## Individual risk model (see 08exercise.jl)

@time begin # 9 seconds approx
    ES = 0.0
    VS = 0.0
    EN = 0.0
    VN = 0.0
    m = 100_000 # number of simulations
    isimS = zeros(m) # vector for simulated total claims
    isimN = zeros(Int, m) # vector for simulated number of claims
    for j ∈ 1:r
        qj = policy.q[j]
        cj = policy.INSAMOUNT[j]
        ES += (1 + k)*cj*qj # Same as ES = ES + (1 + k)*cj*qj
        VS += (cj^2) * qj * (1 + 3k - qj * (1 + k)^2) 
        EN += qj
        VN += qj * (1 - qj)
        Accident = rand(Bernoulli(k), m) # vector of size m
        Death = rand(Bernoulli(policy.q[j]), m) # vector of size m
        isimN = isimN .+ Death
        isimS = isimS .+ (policy.INSAMOUNT[j] .* Death .* (1 .+ Accident))
    end
    println("Theoretical vs simulation approx:")
    println("---------------------------------")
    println("E(S) = ", (ES, mean(isimS)))
    println("V(S) = ", (VS, var(isimS)))
    println("E(N) = ", (EN, mean(isimN)))
    println("V(N) = ", (VN, var(isimN)))
end



## Frequency model (N)

#=
    N = ∑1{Xj>0} = ∑Dj where Dj~Bernoulli(qj), j ∈ {1,…,r},
    with `r`the number of policies in the portfolio. 
    Then Ran N = {0,1,…,r} and E(N) = ∑qj and V(N) = ∑qj(1-qj)
    since D1,…,Dr are independent (but not identically distirbuted).

    Question: Is the distribution of N approximately Binomial?

    In such case E(N) = r⋅qN for some 0 < qN < 1, that is:

    qN = ∑qj / r   (the average of all qj's)

=#

begin 
    qN = mean(policy.q)
    println("Theoretical vs Binomial approx:")
    println("-------------------------------")
    println("E(N) = ", (EN, r*qN))
    println("V(N) = ", (VN, r*qN*(1-qN)))
end

begin # Simulated versus Binomial frequency
    rvN = Binomial(r, qN)
    empN = masaprob(isimN)
    nn = collect(minimum(empN.valores):maximum(empN.valores))
    bar(nn, empN.fmp.(nn), label = "simulated", color = :yellow)
    title!("Frequency"); xaxis!("N"); yaxis!("probability")
    scatter!(nn, pdf(rvN, nn), ms = 3, color = :red, label = "Binomial approx")
end



## Severity model 

#=
    S = Y1 + Y2 + ⋯ + YN

    Given N = n we have to take a sample of size n without replacement
    from the r policies in the portfolio, with probabilities proportional
    to their mortality rate:

    P(policy j) ∝ qj

    and then add their insured amounts Y1,…,Yn (possibly multiplied by 2
    if it was a accident) to get a simulation of S.
=#

function sampleNoReplace(Ω::Vector, p::Vector, n::Integer = length(Ω); warn = true)
    #=
      Generates samples without replacement of size n ≤ r from a 
      set Ω with r elements, and with probabilities p = [p₁,...,pᵣ]
    =#
    ### Input checking
    r = length(Ω)
    if !(isa(Ω, Vector) && isa(p, Vector) && r == length(p))
        @error "`Ω` and `p` must be same size vectors"
        return nothing
    elseif n < 1 || n > r
        @error "`n` must be an integer between 1 and $r"
        return nothing
    end
    if !all(p .≥ 0)
        @error "Values in vector `p` cannot be negative"
        return nothing
    elseif sum(p) ≠ 1.0
        p = p ./ sum(p)
        if warn 
            @warn "Values in `p` where standardized to sum 1.0"
        end
    end
    function simuno(u, probs)
        c = 1
        for θ ∈ cumsum(probs)
            c += θ < u
        end
        return c
    end
    Ω2 = copy(Ω)
    p2 = copy(p)
    S = typeof(Ω)(undef, n)
    for i ∈ 1:n
        S[i] = Ω2[simuno(rand(1)[1], p2)] # simulates first sample element
        k = findfirst(Ω2 .== S[i]) # position of element in Ω2
        nok = setdiff!(collect(1:length(Ω2)), k)
        Ω2 = Ω2[nok] # remove element from Ω2
        p2 = p2[nok] # remove its probability from p2
        p2 = p2 ./ sum(p2) # recalculate remaining probabilities
    end
    return S 
end

# simulate claims frequencies and severities
@time begin # 3 minutes approx for 10,000 simulations
    m = 10_000 # number of simulations
    csimN = rand(rvN, m)
    A = Bernoulli(k)
    csimY = []
    csimS = zeros(m)
    Ω = collect(1:r) # policy numbers j ∈ {1,…,r}
    for i ∈ 1:m
        if csimN[i] > 0
            jj = sampleNoReplace(Ω, policy.q, csimN[i], warn = false)
            sY = policy.INSAMOUNT[jj] .* (1 .+ rand(A, csimN[i]))
            push!(csimY, sY)
            csimS[i] = sum(sY)
        end
    end
end
begin
    println("Individual risk model versus simulated frequency:")
    println("-------------------------------------------------")
    println("E(S) = ", (mean(isimS), mean(csimS)))
    println("M(S) = ", (median(isimS), median(csimS)))
    println("VaR(S) = ", (quantile(isimS, 0.995), quantile(csimS, 0.995)))
end

# frequency versus median severity
begin
    nY = sort(unique(csimN))
    medianY = zeros(length(nY))
    for i ∈ 1:length(nY)
        y = Float64[]
        ni = findall(csimN .== nY[i])
        for t ∈ ni
            append!(y, csimY[t])
        end
        medianY[i] = median(y)
    end
    c = round(cor(nY, medianY), digits = 2)
    scatter(nY, medianY, legend = false, title = "corr = $c", 
            xlabel = "frequency", ylabel = "median severity"
    )
end # --> assume independence between frequency and severity

# Discrete severity model
sY = Float64[]
for v ∈ csimY
    append!(sY, v)
end
sY
length(unique(sY))
sev = masaprob(sY);
scatter(sev.valores, sev.probs, legend = false, xlabel = "severity", ylabel = "probability")
rvY = Categorical(sev.probs)



### Collective risk model

rvN # frequency model
rvY # severity model
sev.valores # severity possible values
m = 100_000 # number of simulations
N = rand(rvN, m)
S = zeros(m)
@time for i ∈ 1:m 
    if N[i] > 0
        S[i] = sum(sev.valores[rand(rvY, N[i])])
    end
end
begin
    println("Individual versus collective risk model:")
    println("----------------------------------------")
    println("E(S) = ", (mean(isimS), mean(S)))
    println("M(S) = ", (median(isimS), median(S)))
    println("VaR(S) = ", (quantile(isimS, 0.995), quantile(S, 0.995)))
end
