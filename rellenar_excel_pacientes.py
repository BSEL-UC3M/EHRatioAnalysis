import os
import pandas as pd
from openpyxl import Workbook

# Ruta base con carpetas PACIENTE 97 a PACIENTE 102
base_dir = "/Users/claudiacastrillonalvarez/Desktop/TIFF_renamed"
pacientes = [f"PACIENTE {i}" for i in range(97, 103)]
tipos = ["PEI", "MRC"]  # Tipos de carpeta internas

# Crear un Excel para cada tipo (MRC y PEI)
for tipo in tipos:
    wb = Workbook()
    wb.remove(wb.active)  # Eliminar hoja por defecto
    output_file = f"/Users/claudiacastrillonalvarez/Desktop/{tipo}_TIFF_Annotations.xlsx"

    for paciente in pacientes:
        ruta_paciente = os.path.join(base_dir, paciente, tipo)
        
        if not os.path.isdir(ruta_paciente):
            print(f"⚠️ No se encontró: {ruta_paciente}")
            continue

        # Obtener nombres de archivos .tif o .tiff
        archivos = sorted([f for f in os.listdir(ruta_paciente) if f.lower().endswith(('.tif', '.tiff'))])

        # Crear DataFrame
        df = pd.DataFrame({
            "File Name": archivos,
            "Annotation": ["" for _ in archivos]
        })

        # Crear hoja con nombre del paciente
        sheet = wb.create_sheet(title=paciente)

        # Escribir encabezados
        for col_idx, column in enumerate(df.columns, start=1):
            sheet.cell(row=1, column=col_idx, value=column)

        # Escribir datos
        for row_idx, row in enumerate(df.itertuples(index=False), start=2):
            for col_idx, value in enumerate(row, start=1):
                sheet.cell(row=row_idx, column=col_idx, value=value)

    # Guardar el archivo Excel
    wb.save(output_file)
    print(f"✅ Excel creado: {output_file}")
