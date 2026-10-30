## Exercise: Bayesian credibility
## Author: Arturo Erdely
## Version date: 2026-10-26

#=
Use the simulations from the collective risk model in file 09exercise.jl
as a prior model for the total claims random variable S, and suppose that
now we have the following observed values of S: 370, 420, 362, 495, 447. 
Use bayesian credibility to obtain updated (posterior) estimations for
E(S), M(S), VaR(S), and SCR(S), and compare to the prior ones.
=#



## Load required packages and code
begin
    using CSV, DataFrames, Distributions, Plots
    include("06EDA.jl")
    include("student.jl")
end


## Read and graph the simulations from the prior model 
begin
    df = CSV.read("S.csv", DataFrame)
    priorS = df.totclaims
    println("Rango = ", extrema(priorS))
    histogram(priorS, normalize = true, label = "prior model", 
              xlabel = "Total claims", ylabel = "density", color = :yellow)
end


## Calibrate hyperparameters and graph the predictive prior
begin
    histogram(priorS, normalize = true, label = "prior model", 
              xlabel = "Total claims", ylabel = "density", color = :yellow)
    μ0 = mean(priorS)
    m0 = mean(priorS)
    q25 = quantile(priorS, 0.25)
    q75 = quantile(priorS, 0.75)
    VaR = quantile(priorS, 0.995)
    SCR = VaR - μ0
    function fobj(θ)
        n0 = θ[1]
        α = θ[2]
        β = θ[3]
        if min(n0, α, β) ≤ 0
            return Inf 
        else
            X = rvStudent3(μ0, 1 / √(n0*α/((n0+1)*β)), 2*α)
            eq1 = (X.qtl(0.25) - q25)^2
            eq2 = (X.qtl(0.75) - q75)^2
            eq3 = (X.qtl(0.995) - VaR)^2
            return eq1 + eq2 + eq3
        end
    end
    priorEDA = EDA(fobj, [0.01, 0.01, 0.01], [100, 100, 100_000])
    @show n0, α, β = priorEDA.x
    x = collect(range(minimum(priorS), maximum(priorS), length = 1_000));
    X = rvStudent3(μ0, 1 / √(n0*α/((n0+1)*β)), 2*α);
    plot!(x, X.pdf.(x), label = "Student predictive prior", lw = 3, color = :blue)
end


## Calculate and graph the predictive posterior
begin
    Sobs = [370,420,362,495,447];
    n = length(Sobs)
    Sobsmean = mean(Sobs)
    Sobsvar = var(Sobs, corrected = false)
    npost = n0 + n
    μpost = (n0*μ0 + n*Sobsmean) / npost
    αpost = α + n/2
    βpost = β + 0.5*n*Sobsvar + 0.5*n0*n*(μ0 - Sobsmean)^2 / npost
    Xpost = rvStudent3(μpost, 1 / √(npost*αpost/((npost+1)*βpost)), 2*αpost+n);
    plot!(x, Xpost.pdf.(x), label = "Student predictive posterior", lw = 3, color = :red)
end


## Compare prior and posterior calculations
begin
    postMean = Xpost.mean
    postMedian = Xpost.median
    postVaR = Xpost.qtl(0.995)
    postSCR = postVaR - postMean
    println("Prior vs posterior, % change")
    println("============================")
    println("Mean: ", round.((μ0, postMean, 100*(postMean/μ0 - 1)), digits = 1))
    println("Median: ", round.((m0, postMedian, 100*(postMedian/m0 - 1))))
    println("VaR: ", round.((VaR, postVaR, 100*(postVaR/VaR - 1))))
    println("SCR: ", round.((SCR, postSCR, 100*(postSCR/SCR - 1))))
end
