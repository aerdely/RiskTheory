### Exercise: Premium principles
### Author: Dr. Arturo Erdely
### Version: 2024-09-11

#=

Consider a portfolio of 1-year term life insurance independent policies from
the file `LIFEinsurance.csv` that specifies age and insured amount for each
policy, with the additional benefit of twice the insured amount in case of
accidental death, assuming that 1 out of 10 deaths is accidental (regardless
of the age). Use the mortality table in the file `mortality.csv` 
   	
Assume that the insurance company collects all the premiums at the beginning
of the year. Consider a low-risk fixed 1-year interest rate of 5% and that
the shareholders of the company expect a ROE (return on equity) of 12%. 
Assuming that deaths are uniformly distributed (probabilistically) throughout
the year, using 100,000 simulations of the portfolio, estimate quantiles
2.5%, 25%, 50% (median), 75%, and 97.5% of the ROE, and the probability
P(ROE ≥ 12%) under each of the following premium principles:


a) Expected value principle.
b) Variance principle.
c) Using formula from Exercise 2.3

Remember that the premiums and the solvency capital requirement must be invested
at the low-risk interest rate of 5% but for the rest of the assets assume that
it is possible to obtain a return of 12%.

=#


### Load needed packages and code

begin
    using CSV, DataFrames, Plots, Distributions
end


### Read data

mort = DataFrame(CSV.File("mortality.csv")) # mortality table
policy = DataFrame(CSV.File("LIFEinsurance.csv")) # insurance policies


#=
   Theoretical mean and variance for total claims E(S) and V(S).
   Add to DataFrame `policy` columns for E(X_j), V(X_j), and 
   D(X_j)=E(max{X_j - E(X_j), 0}).
=#

n = length(policy.AGE) # number of insurance policies
k = 1/10 # probability of accident given death
q = Dict(mort.AGE .=> mort.qx) # mapping age -> qx
policy.EX = zeros(n)
policy.VX = zeros(n)
policy.DX = zeros(n)
ES, VS = 0.0, 0.0
for j ∈ 1:n
    qj = q[policy.AGE[j]]
    cj = policy.INSAMOUNT[j]
    policy.EX[j] = (1 + k)*cj*qj
    ES += policy.EX[j]
    policy.VX[j] = qj*(1-qj)*((1+k)*cj)^2 + qj*(1-k)k*(cj^2)
    VS += policy.VX[j] # assuming independence
    policy.DX[j] = sum(max.((0, cj, 2cj) .- policy.EX[j], 0) .* (1 - qj, qj * (1 - k), qj * k))
end
println("E(S) = ", ES)
println("V(S) = ", VS)
policy[1:5, :]
size(policy)


#=
   Approximate median and 99.5% Value at Risk for total claims: M(S) and $VaR(S).
=#

m = 100_000 # number of simulations
n = length(policy.AGE) # number of insurance policies
S = zeros(m)
@time begin # 30 seconds approx
    Death = zeros(Bool, m, n) 
    Accident = rand(Bernoulli(k), m, n)
    for j ∈ 1:n
        Death[:, j] = rand(Bernoulli(q[policy.AGE[j]]), m)
    end
    for i ∈ 1:m
        S[i] = sum(policy.INSAMOUNT .* Death[i, :] .* (1 .+ Accident[i, :]))
    end
end

Death = 0; Accident = 0; # release memory

begin
    MS = median(S)
    VaR = quantile(S, 0.995)
    println("M(S) = ", MS)
    println("99.5% VaR = ", VaR)
end


#=
  Solvency Capital Requirement (SCR), based on the expected value and the median:
  SCRe(S) and SCRm(S).
=#

begin
    SCRe = VaR - ES # based on expected value
    SCRm = VaR - MS # based on the median
    println("SCRe(S) = ", SCRe)
    println("SCRm(S) = ", SCRm)
end


#=
  Risk Margin, based on the expected value and the median:
  RMe(S) and RMm(S).
=#

begin
    r = 0.12; # shareholders' return rate (ROE)
    i = 0.05; # low-risk return rate
    RMe = (r - i) * SCRe # based on expected value
    RMm = (r - i) * SCRm # based on the median
    println("RMe(S) = ", RMe)
    println("RMm(S) = ", RMm)
end



### PREMIUMS  

## Expected value principle
begin
    θE = RMe / ES
    policy.PErisk = policy.EX
    policy.PEmargin = θE .* policy.EX
    policy.PE = policy.PErisk .+ policy.PEmargin
    policy[1:3, :]
end


## Variance principle
begin
    θV = RMe / VS
    policy.PVrisk = policy.EX
    policy.PVmargin = θV .* policy.VX
    policy.PV = policy.PVrisk .+ policy.PVmargin
    policy[1:3, :]
end


## Comparing the expected value principle and the variance principle:

PEPV = policy.PE ./ policy.PV
quantile(PEPV, [0.025, 0.5, 0.975])
mean(PEPV)

#=
  As expected, individual premiums based on the expected value principle
  are higher (on average and median) that those based on the variance
  principle, since the expected value principle is not consistent and
  loads the risk premium more than it should.
=#
histogram(PEPV, label = "PE / PV", xlabel = "ratio", ylabel = "frequency", color = :yellow)


## Exercise 2.3 formula

begin
    sD = sum(policy.DX)
    policy.PDrisk = (MS / ES) .* policy.EX
    policy.PDmargin = (RMm / sD) .* policy.DX
    policy.PD = policy.PDrisk .+ policy.PDmargin
    policy[1:3, 9:14]
end 

sum(policy.PD), MS + RMm # checking the sum of total premiums

# comparing against the variance principle:
PDPV = policy.PD ./ policy.PV
quantile(PDPV, [0.025, 0.5, 0.975])
mean(PDPV)
histogram(PDPV, label = "PD / PV", xlabel = "ratio", ylabel = "frequency", color = :green)


### Estimation of P(ROE ≥ 12%)

#   ... pending