### Exercise: Ruin Theory
### Author: Dr. Arturo Erdely
### Version date: 2026-11-10


## Packages

using Distributions, Plots, StatsBase


## Read and process data

countlines("siniestros.csv") 
contenido = readlines("siniestros.csv")

begin
    m = length(contenido)
    sev = fill(Float64[], m) # [Y1, Y2, YN]
    frec = zeros(Int, m) # N
    reclam = zeros(m) # S
    for i ∈ 1:m
        v = parse.(Float64, split(contenido[i], ','))
        if v == [0.0]
            sev[i] = []
        else
            sev[i] = v
            frec[i] = length(v)
            reclam[i] = sum(v)
        end
    end
end

sev	# vector of vectors of severities per period	
frec	# vector of frequency per period
reclam	# vector of total claims per period

function resumen(x) # summarize a given vector
    (median = median(x), mean = mean(x), quartile3 = quantile(x, 0.75), VaR = quantile(x, 0.995))
end

resumen(reclam)


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
    plot(legend = false, title = "Severity vs frequency")
    xaxis!("number of claims per period")
    yaxis!("claim amount")
    for k ∈ frecvalores
        ifrec = findall(frec .== k)
        for ℓ ∈ ifrec
            scatter!(fill(k, k), sev[ℓ], color = :gray)
        end
    end
    current()
    savefig("z03severity.png")
end

begin
    medias = zeros(length(frecvalores))
    desvst = zeros(length(frecvalores))
    plot(legend = false, title = "Log-severity vs frequency")
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
    plot(frecvalores, medias, color = :red, lw = 3, legend = false)
    title!("Mean log-severity conditional on frequency")
    xaxis!("frequency")
    p1 = yaxis!("mean")
    savefig("05logmeanseverity.png")
    plot(frecvalores, desvst, color = :blue, lw = 3, legend = false)
    title!("Standard deviation of conditional log-severity")
    xaxis!("frequency")
    p2 = yaxis!("standard deviation")
    savefig("06logsevstddev.png")
    plot(p1, p2, layout = (2,1), size = (600,600))
end
