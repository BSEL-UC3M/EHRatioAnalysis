import os

# Ruta base donde están todas las carpetas de pacientes
directorio_base = '/Users/claudiacastrillonalvarez/Desktop/patients_tiff_104_107'

# Pacientes a procesar (ejemplo: del 104 al 107)
pacientes = ["PACIENTE_107"]
# Subcarpetas esperadas
tipos = ["MRC", "PEI"]

# Recorremos cada paciente
for paciente in pacientes:
    ruta_paciente = os.path.join(directorio_base, paciente)

    if not os.path.isdir(ruta_paciente):
        print(f"❌ No se encontró la carpeta del paciente: {ruta_paciente}")
        continue

    num_paciente = paciente.split("_")[1]  # Extrae el número, ej. '104'

    for tipo in tipos:
        ruta_tipo = os.path.join(ruta_paciente, tipo)

        if not os.path.isdir(ruta_tipo):
            print(f"⚠️  Falta la subcarpeta {tipo} en {ruta_paciente}")
            continue

        for archivo in os.listdir(ruta_tipo):
            ruta_archivo = os.path.join(ruta_tipo, archivo)

            # Aceptar tanto .tif como .tiff
            if os.path.isfile(ruta_archivo) and archivo.lower().endswith((".tif", ".tiff")):
                nombre_imagen = os.path.splitext(archivo)[0]
                extension = os.path.splitext(archivo)[1]  # conserva .tif o .tiff

                nuevo_nombre = f"{tipo}_{num_paciente}_{nombre_imagen}{extension}"
                ruta_nuevo_nombre = os.path.join(ruta_tipo, nuevo_nombre)

                try:
                    os.rename(ruta_archivo, ruta_nuevo_nombre)
                    print(f"✅ Renombrado: {archivo} -> {nuevo_nombre}")
                except Exception as e:
                    print(f"❌ Error al renombrar {archivo}: {e}")
