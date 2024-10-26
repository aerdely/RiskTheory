### Método ABC para inferencia bayesiana
### Autor: Dr. Arturo Erdely
### Versión: 2024-10-13


# Cargar paquetes y código externo necesarios:
begin
    using Random # no requiere instalación previa 
    using Distributions, Plots, LaTeXStrings # previamente instalados
    include("02probestim.jl") # en misma carpeta
end


# Simulamos una muestra aleatoria
begin
    r, θ = 17, 0.35 # parámetros
    X = Binomial(r, θ) # variable aleatoria Binomial
    n = 300 # tamaño de muestra a simular
    Random.seed!(123) # para reproducibilidad de este ejemplo
    muestra = rand(X, n)
    mm = maximum(muestra)
    println("Muestra:\n", muestra)
    println("Máximo = ", mm)
end


# Distribución a priori
function priori(a::Int, b::Int, nsim::Int)
    ab = collect(a:b)
    k = length(ab)
    pp = fill(1/k, k)
    r = rand(Categorical(pp), nsim) .+ a .- 1
    θ = rand(nsim) # simula Uniforme(0,1)
    return (r = r, θ = θ)
end


# Simula a priori
begin
    nsim = 1_000_000
    p = priori(mm, 3*mm, nsim)
    δ = zeros(nsim)
    muestraord = sort(muestra)
    for i ∈ 1:nsim
        muestraABC = rand(Binomial(p.r[i], p.θ[i]), n)
        δ[i] = sum(abs.(sort(muestraABC) - muestraord))
    end
end


# Método ABC
begin
    nselec = 300
    iselec = sortperm(δ)[1:nselec]
    post = (r = p.r[iselec], θ = p.θ[iselec])
    scatter(post.r, post.θ, label = "simulaciones a posteriori")
    xaxis!(L"r|\mathbf{x}"); yaxis!(L"θ|\mathbf{x}")
    scatter!([r], [θ], color = :red, ms = 6, label = "r = $r, θ = $θ")
end


# Inferencias marginales a posteriori
begin
    rpost = masaprob(post.r)
    plot(rpost.valores, rpost.probs, color = :gray, label = "")
    scatter!(rpost.valores, rpost.probs, label = "")
    vline!([r], label = "r = $r")
    xaxis!(L"r"); yaxis!(L"p(r|\mathbf{x})")
end

begin
    θpost = densprob(post.θ)
    x = range(θpost.min, θpost.max, length = 1_000)
    plot(x, θpost.fdp.(x), lw = 3, label = "")
    vline!([θ], label = "θ = $θ")
    xaxis!(L"\theta"); yaxis!(L"p(\theta|\mathbf{x})")
end
