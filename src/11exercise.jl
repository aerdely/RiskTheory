### Exercise: Collective Risk Model
### Author: Dr. Arturo Erdely
### Version: 2024-09-04

#=

Consider the data in the file `CRMdata.txt` which represents weekly claims
(in million pesos) in the last 20 years for a certain insurance product, 
one week per row. It is only data for those weeks were claims were filed. 
Fit a collective risk model and estimate the expected value, variance, 
median, and 99.5% Value at Risk (VaR) of the total claims per week.

=#


### Load needed packages and code

begin
    using CSV, DataFrames, Plots, Distributions, StatsBase
    include("02probestim.jl")
end



### Read and prepare data for analysis

begin
    f = open("CRMdata.txt") # open connection to file
    rawdata = readlines(f) # read line by line from connection
    close(f) # close connection, always very important!
    rawdata # display vector of strings
end

rawdata[1]

split(rawdata[1], ",") # split string using "," a separator

parse.(Float64, split(rawdata[1], ",")) # convert strings to Float64 numbers

# create a vector of vectors of claims
begin
    data = []
    for i ∈ 1:length(rawdata)
        push!(data, parse.(Float64, split(rawdata[i], ",")))
    end
    data
end



### Frequency modeling

## Estimation of N|N>0

Ndata = length.(data) # counts of N given N>0 
Npos = masaprob(Ndata); # conditional pmf N|N>0
[Npos.valores Npos.probs]
ENpos = mean(Ndata) # E(N|N>0)
VNpos = var(Ndata) # V(N|N>0)

begin
    nn = collect(Npos.valores[begin]:Npos.valores[end])
    pnn = Npos.fmp.(nn)
    bar(nn, pnn, xlabel = "Frequency N | N > 0", ylabel = "P(N = n | N > 0)", legend = false)
end

# There are 
52*20 # weeks in 20 years
# and just in 
length(data) # weeks claims were files
# therefore in 
52*20 - length(data) # weeks there were no claims
PN0 = (52*20 - length(data)) / (52*20) # P(N=0)



## Fitting a model for N-1|N>0

ENpos0 = mean(Ndata .- 1) # E(N-1|N>0) = E(N|N>0) - 1
VNpos0 = var(Ndata .- 1) # V(N-1|N>0) = V(N|N>0)
# Since ENpos0 < VNpos we may try to fit a Negative Binomial (r,p)
# Method of moments estimation (just to make it quickly)
# If N ~ NegBinom(r,p) then E(N) = r(1-p)/p and V(N) = E(N)/p
# therefore p = E(N)/V(N) and r = pE(N)/(1-p)
pmom = ENpos0 / VNpos0
rmom = pmom * ENpos0 / (1 - pmom)
NB = NegativeBinomial(pmom, rmom)
# comparing empirical vs fitted model
Npos.fmp(1), pdf(NB, 1-1) # P(N = 1 | N > 0)
Npos.fmp(2), pdf(NB, 2-1) # P(N = 2 | N > 0)
1 - sum(Npos.fmp.(1:10)), 1 - cdf(NB, 9) # P(N > 10 | N > 0)
begin
    nn = collect(Npos.valores[begin]:Npos.valores[end])
    pnn = Npos.fmp.(nn)
    bar(nn, pnn, xlabel = "Frequency N | N > 0", ylabel = "P(N = n | N > 0)", label = "Empirical")
    scatter!(nn, pdf(NB, nn .- 1), label = "Negative Binomial", ms = 2)
end
# --> Not a good fit


## Fitting a model for N with Ran N = {0,1,…}

[Npos.valores Npos.probs] # pmf for N|N>0
PN0 = (52*20 - length(data)) / (52*20) # P(N=0)
#=
  P(N=n) = P(N=n, N>0) + P(N=n, N=0)
         = P(N=n|N>0)P(N>0) + P(N=n|N=0)P(N=0)
         = P(N=n|N>0)[1-P(N=0)] + P(N=0)⋅1{n=0}
=#
pmfN(n::Integer) = Npos.fmp(n)*(1-PN0) + PN0*(n==0)
Ndata0 = copy(Ndata)
append!(Ndata0, zeros(Int, 52*20 - length(data))) # adding zeros
μ = mean(Ndata0)
σ2 = var(Ndata0)
pmom = μ / σ2
rmom = μ*pmom / (1 - pmom)
NB = NegativeBinomial(pmom, rmom)
# comparing empirical vs fitted model
pmfN(0), pdf(NB, 0) # P(N = 0)
pmfN(1), pdf(NB, 1) # P(N = 1)
pmfN(2), pdf(NB, 2) # P(N = 1)
1 - sum(pmfN.(collect((0:10)))), 1 - cdf(NB, 10) # P(N > 10)
begin
    nn = collect(0:Npos.valores[end])
    pnn = pmfN.(nn)
    bar(nn, pnn, xlabel = "Frequency N", ylabel = "P(N = n)", label = "Empirical")
    scatter!(nn, pdf(NB, nn), label = "Negative Binomial", ms = 2)
end
# --> Negative Binomial not a bad fit



### Severity modeling

## Check if Y1,Y2,… independent

Y1 = Float64[]
Y2 = Float64[]
for i ∈ 1:length(data)
    m = length(data[i])
    if m ≥ 2
        for j in 1:(m-1)
            push!(Y1, data[i][j])
            push!(Y2, data[i][j+1])
        end
    end
end
data[1:3]
[Y1[1:7] Y2[1:7]]
begin
    c = round(corspearman(Y1,Y2), digits = 4)
    scatter(Y1, Y2, xlabel = "Y(t)", ylabel = "Y(t+1)", title = "corr = $c", legend = false)
end
begin
    ratio = Y2 ./ Y1
    mr = round(median(ratio), digits = 2)
    plot(ratio, ylabel = "ratio = Y(t+1) / Y(t)", label = "ratio", title = "Median ratio = $mr")
    hline!([1.0], label = "equality (1.0)")
end
# --> Assume independent claims


## Check for frequency vs severity independence

Nunique = sort(unique(Ndata))
medianY = zeros(length(Nunique))
for i ∈ 1:length(Nunique) 
    idn = findall(Ndata .== Nunique[i])
    Y = Float64[]
    for k ∈ idn
        Y = vcat(Y, data[k])
    end
    medianY[i] = median(Y)
end
scatter(Nunique, medianY, legend = false, xlabel = "Frequency", ylabel = "Median severity")

function reglin(x::Vector{<:Real}, y::Vector{<:Real})
    # simple linear regression: y = a + bx
    if length(x) ≠ length(y)
        error("vectors must have the same length")
        return nothing
    end
    n = length(x)
    sxy = sum(x .* y)
    sx2 = sum(x .^ 2)
    mx = sum(x) / n
    my = sum(y) / n
    b = (sxy - n*mx*my) / (sx2 - n*mx^2)
    a = my - b*mx
    return(a, b)
end

a, b = reglin(Nunique, medianY)
n = collect(minimum(Nunique):maximum(Nunique))
plot!(n, a .+ b.*n, lw = 2, color = :red)
# --> Let's assume it is not quite significant dependence


## Analyze severity data

Y = Float64[]
# join all severity values into a single vector
for d ∈ data 
    Y = vcat(Y, d)
end
length(Y), sum(Ndata) # should be the same
allunique(Y)
length(unique(Y)) / length(Y)
# --> let's assume a continuous rv model for severity
sev = densprob(Y);
yy = collect(range(sev.min, sev.max, length = 1_000));
plot(yy, sev.fdp.(yy), lw = 2, legend = false, xlabel = "Severity", ylabel = "density")
# --> too heavy and long tails, let's transform the data with log

logY = log.(Y)
logsev = densprob(logY);
logyy = collect(range(logsev.min, logsev.max, length = 1_000));
plot(logyy, logsev.fdp.(logyy), lw = 2, label = "empirical", xlabel = "Log-severity", ylabel = "density")


## Try Pareto density for log-severity

rvLogY = fit_mle(Pareto, logY)
plot!(logyy, pdf(rvLogY, logyy), lw = 2, color = :red, label = "Pareto")



### Collective risk model (under usual asssumptions)

# S = Y1 + ⋯ + YN

NB # Frequency (N)
rvLogY # Log-severity (log Y)
PS0 = PN0 # P(S = 0) = P(N = 0)
Sdata = sum.(data) # S|N>0
Sdata0 = copy(Sdata)
append!(Sdata0, zeros(52*20 - length(data))) # adding zeros

function simulateCRM(m = 1_000_000)
    # m = number of simulations
    S = zeros(m)
    N = rand(NB, m)
    iN = findall(N .≥ 1)  # positions in N such that N ≥ 1
    Y = []
    for i ∈ iN
        logYi = rand(rvLogY, N[i]) # logY1,…,logYN
        Yi = exp.(logYi)
        S[i] = sum(Yi)
        append!(Y, Yi)
    end
    PS0 = 1 - count(S .> 0)/m # <-- P(S = 0)
    return (ES = mean(S), MS = median(S), VS = var(S), PS0 = PS0,
            VaRS = quantile(S, 0.995), S = S, N = N, Y = Y
    )
end

# execute several times
begin
    sim = simulateCRM(52*20)
    println("Observed versus simulated")
    println("-------------------------")
    println("E(S) = ", (mean(Sdata0), sim.ES)) # too heavy tails --> E(S) = ∞
    println("M(S) = ", (median(Sdata0), sim.MS))
    println("VaR(S) = ", (quantile(Sdata0, 0.995), quantile(sim.S, 0.995)))
end

begin
    sim = simulateCRM()
    println("Observed versus simulated")
    println("-------------------------")
    println("E(S) = ", (mean(Sdata0), sim.ES)) # too heavy tails --> E(S) = ∞
    println("M(S) = ", (median(Sdata0), sim.MS))
    println("VaR(S) = ", (quantile(Sdata0, 0.995), quantile(sim.S, 0.995)))
end


# Requires further analysis, too many questionable assumptions...
