### Exercise: Premium principles
### Author: Dr. Arturo Erdely
### Version: 2024-09-12

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
   Approximate median and 99.5% Value at Risk for total claims: M(S) and VaR(S).
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

## Claims

function simulateClaims() # from each insurance policy
    n = length(policy.AGE)
    k = 1/10 # probability of accident given death
    Death = zeros(n) 
    Accident = rand(Bernoulli(k), n)
    Claim = zeros(n)
    for j ∈ 1:n
        Death[j] = rand(Bernoulli(q[policy.AGE[j]]), 1)[1]
        Claim[j] = policy.INSAMOUNT[j] * Death[j] * (1 + Accident[j])
    end
    return Claim # vector of each policy claim (if any)
end

ES, VaR, sum(simulateClaims()) # try several times just to see


## Accouting 

function simulateAcc(premiumPrinciple::String) 
    # "E" = expected value | "V" = variance | "D" = Exercise 2.3
    # Liabilities
    BEL = zeros(367); RM = zeros(367); L = zeros(367)
    # Assets: AR (reserves -> rate i), AF (free -> r)
    AR = zeros(367); AF = zeros(367); A = zeros(367) 
    # Capital
    SCR = zeros(367); K = zeros(367) 
    # [1] -> starting values | [2:366] -> 365 days of the year | [367] -> closing of the year
    if premiumPrinciple ∈ ["E", "V"]
        BEL[1] = ES; RM[1] = RMe; SCR[1] = SCRe
    elseif premiumPrinciple == "D"
        BEL[1] = MS; RM[1] = RMm; SCR[1] = SCRm
    else
        println("Unknown premium principle, try again!")
        return nothing
    end
    # starting Balance Sheet:
    L[1] = BEL[1] + RM[1]
    K[1] = SCR[1]
    AR[1] = L[1] + SCR[1]; AF[1] = 0; A[1] = AR[1] + AF[1]
    # claims of the year:
    claims = simulateClaims() # from each insurance policy
    idClaims = findall(a -> a > 0, claims) # policies with positive claims
    n = length(claims) # number of insurance policies in portfolio
    dayClaim = rand(DiscreteUniform(2, 366), n) # uniform distribution of claims during the year
    C = zeros(367) # to allocate total claims amount per day
    # equivalent daily rates:
    ii = (1 + i)^(1/365) - 1 # equivalent daily rate
    rr = (1 + r)^(1/365) - 1 # equivalent daily rate
    # simulate daily accounting:
    for d ∈ 2:366
        policies = findall(j -> j == d, dayClaim) ∩ idClaims
        C[d] = sum(claims[policies])
        BEL[d] = max(0, BEL[d-1] - C[d]) # until we run out of BEL
        RM[d] = max(0, RM[d-1] + min(0, BEL[d-1] - C[d])) # until we run out of RM (after BEL)
        L[d] = BEL[d] + RM[d]
        SCR[d] = max(0, SCR[d-1] + min(0, BEL[d-1] + RM[d-1] - C[d])) # until we run out of SCR (after RM)
        A[d] = (1 + ii)*AR[d-1] + (1 + rr)*AF[d-1] - C[d]
        AR[d] = L[d] + SCR[d]
        AF[d] = A[d] - AR[d]
        K[d] = A[d] - L[d]
    end
    # closing of the year:
    K[367] = K[366] + BEL[366] + RM[366] # kill remaining BEL and RM (if any)
    AR[367] = 0; AF[367] = A[366]; A[367] = A[366] # all assets become free
    C[367] = sum(C[2:366]) # total claims paid
    results = (C = C, AR = AR, AF = AF, A = A, BEL = BEL, RM = RM, L = L, SCR = SCR, K = K)
    return results
end

# for example:

s = simulateAcc("D")
# 1 -> initial values | 366 -> end of year | 367 -> closing of the year
y = [1, 366, 367]
df = DataFrame(C = s.C[y], AR = s.AR[y], AF = s.AF[y], A = s.A[y], BEL = s.BEL[y],
               RM = s.RM[y], L = s.L[y], SCR = s.SCR[y], K = s.K[y]
)
print("Total claims paid = ", round(s.C[367], digits = 2))
println("   versus point estimate = ", round(s.BEL[1], digits = 2))
println("Return on Equity (ROE) = ", round(100 * (s.K[367] / s.K[1] - 1), digits = 2), "%")


## Return on Equity (ROE)

function simulateROE(premiumPrinciple::String, numsims = 100_000)
    if premiumPrinciple ∉ ["E", "V", "D"]
        println("Premium principle must be E, V, or D. Try again!")
        return nothing
    end
    ROE = zeros(numsims)
    for j ∈ 1:numsims
        s = simulateAcc(premiumPrinciple)
        ROE[j] = 100 * (s.K[367] / s.K[1] - 1)
    end
    return ROE
end

# Under the expected value principle

@time ROE_E = simulateROE("E", 100_000); # aprox 13 min 

quantile(ROE_E, [0.025, 0.25, 0.50, 0.75, 0.975])

begin
    println("P(ROE_E ≥ 12%) ≈ ", mean(ROE_E .≥ 12)) 
    println("P(insolvency) = P(ROE_E < -100%) ≈ ", mean(ROE_E .< -100))
    histogram(ROE_E, color = :yellow, label = "", xlabel = "ROE_E", ylabel = "frequency")
    vline!([12], color = :blue, lw = 2, label = "target")
    vline!([-100], color = :red, lw = 2, label = "insolvency")
end

# Under the variance principle 

@time ROE_V = simulateROE("V", 100_000); # aprox 13 min 

quantile(ROE_V, [0.025, 0.25, 0.50, 0.75, 0.975])

begin
    println("P(ROE_V ≥ 12%) ≈ ", mean(ROE_V .≥ 12)) 
    println("P(insolvency) = P(ROE_V < -100%) ≈ ", mean(ROE_V .< -100))
    histogram(ROE_V, color = :yellow, label = "", xlabel = "ROE_V", ylabel = "frequency")
    vline!([12], color = :blue, lw = 2, label = "target")
    vline!([-100], color = :red, lw = 2, label = "insolvency")
end

# Under the fomula from exercise 2.3

@time ROE_D = simulateROE("D", 100_000); # aprox 13 min 

quantile(ROE_D, [0.025, 0.25, 0.50, 0.75, 0.975])

begin
    println("P(ROE_D ≥ 12%) ≈ ", mean(ROE_D .≥ 12)) 
    println("P(insolvency) = P(ROE_D < -100%) ≈ ", mean(ROE_D .< -100))
    histogram(ROE_D, color = :yellow, label = "", xlabel = "ROE_D", ylabel = "frequency")
    vline!([12], color = :blue, lw = 2, label = "target")
    vline!([-100], color = :red, lw = 2, label = "insolvency")
end
