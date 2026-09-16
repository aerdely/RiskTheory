# Autor: Arturo Erdely
# Fecha: 15-SEP-2026

"""
    vi(archivo::String, num::Int = 1, ini::Int = 1, fin::Int = 0)

Mostrar en la terminal el contenido de `archivo` numerando cada línea, comenzando por la línea `ini` (la primera, si no se especifica) y terminando en la línea `fin` (o la última, si no se especifica). Si `num = 1` (default) se agrega numeración.
"""
function vi(archivo::String, num::Int = 1, ini::Int = 1, fin::Int = 0)
	c = collect(enumerate(readlines(archivo)))
	if fin == 0
		fin = length(c)
	end
	if num == 1
		for i in ini:fin
			println(string(c[i][1]), "  ", c[i][2])
		end
	else
		for i in ini:fin
			println(c[i][2])
		end
	end
	return nothing
end
