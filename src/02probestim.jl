### Estimación empírica de probabilidades
### Autor: Dr. Arturo Erdely
### versión: 2025-12-28

### Funciones:  distprob  masaprob  densprob  modaDiscreta  modaContinua  intervalo


"""
    distprob(muestra::Vector{<:Real})

Estima la función de distribución acumulativa `F(x) = P(X ≤ x)` de una
variable aleatoria `X` a partir de un vector `muestra` de observaciones de `X`.
Entrega una tupla etiquetada con los siguientes nombres:
- `fda` la función de distribución acumulativa estimada
- `min`, `max` mínimo y máximo muestrales
- `mord` la muestra ordenada de menor a mayor

## Ejemplo
```
muestra = randn(10_000);
D = distprob(muestra);
D.min, D.max
D.mord
F = D.fda;
F(0)
diff(F.([-1.96, 1.96]))
```
"""
function distprob(muestra::Vector{<:Real})
    Fn(x) = count(muestra .≤ x) / length(muestra)
    inf, sup = extrema(muestra) # mínimo y máximo muestrales
    return (fda = Fn, min = inf, max = sup, mord = sort(muestra))
end


"""
    masaprob(muestra::Vector)

Estima la función de masa de probabilidades a partir del vector `muestra`.
Entrega una tupla etiquetada con los siguientes nombres:
- `fmp` una función que estima la probabilidad de su argumento
- vector con los `valores` distintos encontrados en la `muestra`
- `probs` vector de proporciones muestrales de los `valores`
- la `muestra` proporcionada

## Ejemplo
```
muestra = rand(['a','b','c'], 100);
M = masaprob(muestra);
hcat(M.valores, M.probs) # tabla
M.fmp('c')
M.fmp.(['a', 'b'])
sum(M.fmp.(M.valores)) # deber ser 1.0 (aprox)
```
"""
function masaprob(muestra::Vector)
    valores = unique(muestra) # valores distintos de la muestra
    try 
        sort!(valores) # intenta ordenar valores
    catch
        @warn "Conjunto no ordenable" # avisa si no se pudo
    end 
    nv = length(valores) # número de valores distintos en la muestra
    frecuencia = zeros(Int, nv) # vector de frecuencias absolutas
    n = length(muestra) # tamaño de la muestra
    for i ∈ 1:nv # calcular frecuencia absoluta de los distintos valores de la muestra
        for j ∈ 1:n
            if valores[i] == muestra[j]
                frecuencia[i] += 1
            end
        end
    end
    proporción = frecuencia ./ n
    d = Dict(valores .=> proporción)
    function f(x)
        if x ∈ valores
            return d[x]
        else
            return 0.0
        end
    end
    return (fmp = f, valores = valores, probs = proporción, muestra = muestra)
end


"""
    modaDiscreta(muestra::Vector)   

Estima la(s) moda(s) a partir de un vector `muestra` de valores observados de una variable aleatoria discreta. 
Devuelve una tupla etiquetada con:
- `moda`: vector con la(s) moda(s)
- `prob`: probabilidad empírica de la(s) moda(s)
- `moda2`: vector con el(los) segundo(s) valor(es) empíricamente más probable(s)
- `prob2`: probabilidad empírica del(los) segundo(s) valor(es) más probable(s)

## Ejemplo
```
muestra = [1,0,4,0,3,5,6,3,0,3,1,6];
modas = modaDiscreta(muestra);
modas
```
"""
function modaDiscreta(muestra::Vector)
    n = length(muestra)
    M = masaprob(muestra)
    probsord = sort(unique(M.probs); rev = true)
    modas = M.valores[findall(M.probs .== probsord[1])]
    probmodas = M.fmp(modas[1])
    if length(modas) < n # hay segundo valor más probable
        segval = M.valores[findall(M.probs .== probsord[2])]
        probsegval = M.fmp(segval[1])
    else
        segval = []
        probsegval = Float64[]
    end
    return (moda = modas, prob = probmodas, moda2 = segval, prob2 = probsegval)
end



"""
    densprob(muestra::Vector{<:Real}, nclases::Integer = 0)

Estima la función de densidad correspondiente al vector `muestra` de observaciones
de una variable aleatoria absolutamente continua, mediante una función constante
por intervalos, considerando un total de `nclases` intervalos, parámetro que de
omitirse se calcula como el `mín{√n, 30}` donde `n` es el tamaño de muestra.
Entrega una tupla etiquetada con los siguientes nombres:
- `fdp` es la función de densidad estimada
- `clases` es un vector que particiona el rango observado en `nclases`
- `reps` es un vector de puntos medios de las clases (representantes)
- `min` y `max` son el mínimo y máximo muestrales
- `nclases` el número de clases o intervalos de igual longitud
- y la `muestra` utilizada

## Ejemplo
```
muestra = randn(10_000); # Normal(0,1)
D = densprob(muestra, 25);
D.fdp(0)
D.fdp.([-2,-1,0,1,2])
D.min, D.max
D.clases
D.reps
D.nclases
sum(D.fdp.(D.reps)) * (D.max - D.min)/D.nclases # debe ser 1.0 (aprox)
D.muestra
```
"""
function densprob(muestra::Vector{<:Real}, nclases::Integer = 0)
    # muestra = vector de valores observados unidimensionales
    # numint = número de intervalos (clases) a considerar
    n = length(muestra) # tamaño de muestra
    if nclases == 0 # valor por defecto
        m = min(Int(round(sqrt(n))), 30)
    else
        m = nclases
    end    
    inf, sup = extrema(muestra) # mínimo y máximo muestrales
    clases = collect(range(inf, sup, length = m + 1)) # extremos de intervalos de clase
    repsclase = (clases[1:end-1] + clases[2:end])/2 # puntos medios de las clases
    function f(x::Real) # función de densidad estimada
        if x < inf || x > sup
            return 0.0 # por estar fuera del rango observado
        else
            i = count(x .≥ clases) # determinar clase a la que pertenece x
            if i ∈ [m, m+1] # último intervalo es de la forma [a,b]
                a = clases[m]
                b = clases[m+1]
                conteo = count(a .≤ muestra .≤ b)
            else # los demás intervalos son de la forma [a,b)
                a = clases[i]
                b = clases[i+1]
                conteo = count(a .≤ muestra .< b)
            end
            return conteo / (n*(b-a))
        end
    end
    return (fdp = f, clases = clases, reps = repsclase, min = inf, max = sup, nclases = m, muestra = muestra)
end


"""
    modaContinua(muestra::Vector{<:Real}, nclases::Integer = 0)

Estima la moda a partir de un vector `muestra` de valores observados de una variable aleatoria continua,
considerando un total de `nclases` intervalos, parámetro que de omitirse se calcula como el `mín{√n, 30}`
donde `n` es el tamaño de muestra. Devuelve una tupla con:
- `moda`: vector con la(s) moda(s) empírica(s)
- `dens`: valor de la densidad empírica en la moda empírica
- `moda2`: vector con el(los) segundo(s) valor(es) con mayor densidad empírica
- `dens2`: valor de la densidad empírica en el(los) segundo(s) valor(es) con mayor densidad empírica

## Ejemplo
```
z = randn(10_000);
muestra = vcat(z .- 2.0, z .+ 2.0); # mezcla de dos normales
modas = modaContinua(muestra, 50);  
modas
```
"""
function modaContinua(muestra::Vector{<:Real}, nclases::Integer = 0)
    D = densprob(muestra, nclases)
    intdens = sort(D.fdp.(D.reps); rev = true)
    moda = D.reps[findall(D.fdp.(D.reps) .== intdens[1])]
    densmoda = intdens[1]
    if length(moda) < length(D.reps) # hay segundo valor más probable
        segval = D.reps[findall(D.fdp.(D.reps) .== intdens[2])]
        densegval = intdens[2]
    else
        segval = Float64[]
        densegval = Float64[]
    end
    return (moda = moda, dens = densmoda, moda2 = segval, dens2 = densegval)
end

"""
    intervalo(muestra::Vector{<:Real}, prob = 0.95)

    Estima el intervalo de probabilidad mínima (de longitud mínima) que contiene una
proporción `prob` de la muestra dada. Devuelve una tupla etiquetada con:

- `int`: el intervalo estimado  
- `tam`: la longitud del intervalo estimado
- `prob`: la proporción de la muestra contenida en el intervalo
- `minmax`: el mínimo y máximo muestrales
- `ran`: el rango muestral
## Ejemplo
```
muestra = randn(100_000);
intervalo(muestra, 0.95)
```
"""
function intervalo(muestra::Vector{<:Real}, prob = 0.95)
    xx = sort(muestra)
    n = length(xx)
    k = Int(round(prob * n))
    minlen = Inf
    a = xx[1]
    b = xx[k]
    for i ∈ 1:(n - k + 1)
        len = xx[i + k - 1] - xx[i]
        if len < minlen
            minlen = len
            a = xx[i]
            b = xx[i + k - 1]
        end
    end
    return (int = [a, b], tam = b-a, prob = k/n, minmax = [xx[1], xx[end]], ran = xx[end] - xx[1])
end


@info "distprob  masaprob  densprob  modaDiscreta  modaContinua  intervalo"
