# Autor: Arturo Erdely
# Fecha: 16-SEP-2026

"""
    vi(archivo::AbstractString; num::Bool = true, ini::Int = 1, fin::Int = 0)

Mostrar en la terminal el contenido de `archivo` línea por línea, comenzando por la línea `ini` (la primera, si no se especifica) y terminando en la línea `fin` (o la última, si no se especifica). Si `num = true` (default) se agrega numeración a cada línea.
"""
function vi(archivo::AbstractString; num::Bool = true, ini::Int = 1, fin::Int = 0)

    lineas = readlines(archivo)
    n = length(lineas)

    fin = fin == 0 ? n : min(fin, n)
    ini = max(ini, 1)

    ancho = ndigits(fin)

    for i in ini:fin
        num ? println(lpad(string(i), ancho), "  ", lineas[i]) : println(lineas[i])
    end

    return nothing
end