### Exercise: Classical credibility
### Author: Dr. Arturo Erdely
### Version: 2026-10-26

#=

Consider the data in the file `CRMdata.txt` which represents weekly claims
(in million pesos) in the last 20 years for a certain insurance product, 
one week per row. It is only data for those weeks were claims were filed. 
=#


### Load needed packages and code

using CSV, DataFrames, Plots, Distributions



### Read and prepare data

begin
	rawdata = readlines("CRMdata.txt") # read data from file
	# create a vector of vectors of claims
	data = []
	for i ∈ 1:length(rawdata)
        	push!(data, parse.(Float64, split(rawdata[i], ",")))
	end
	data
end

# There are 
52*20 # weeks in 20 years
# and just in 
length(data) # weeks claims were filed
# therefore in 
52*20 - length(data) # weeks there were no claims

Sdata = sum.(data) # S|N>0
Sdata0 = copy(Sdata);
append!(Sdata0, zeros(52*20 - length(data))) # adding zeros

ES = mean(Sdata0)
VS = var(Sdata0)
nS = length(Sdata0)


## Full credibility

fullcred_cheby(β, p, ES, VS) = Int(ceil(VS / ((1-p)*(β^2)*(ES^2))))

fullcred_normal(β, p, ES, VS) = Int(ceil(((quantile(Normal(0,1), (1+p)/2))^2) * VS / ((β^2)*(ES^2))))

begin
	β = 0.05
	p = 0.90
	mfC = fullcred_cheby(β, p, ES, VS)
	mfN = fullcred_normal(β, p, ES, VS)
	println("Full credibility")
	println("----------------")
	println("  Chebyshev = $(mfC) periods")
	println("     Normal = $(mfN) periods")
	println()
	println("We have $(nS) observed periods")
end


## Partial credibility

partialcred_cheby(β, p, nS, ES, VS) = min(1, β*ES*sqrt(nS*(1-p))/sqrt(VS))

partialcred_normal(β, p, nS, ES, VS) = min(1, β*ES*sqrt(nS)/(quantile(Normal(0,1), (1+p)/2)*sqrt(VS)))

begin
	β = 0.05
	p = 0.90
	α_cheby = partialcred_cheby(β, p, nS, ES, VS)
	α_normal = partialcred_normal(β, p, nS, ES, VS)
	println("Partial credibility")
	println("-------------------")
	println("  Chebyshev = $(α_cheby * 100) %")
	println("     Normal = $(α_normal * 100) %")
end
