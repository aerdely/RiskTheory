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


## Load needed packages and code

begin
    using CSV, DataFrames, Plots, Distributions, StatsBase
    include("02probestim.jl")
end



## Read and prepara data for analysis

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



## Frequency modeling: pending.


## Severity modeling: pending.