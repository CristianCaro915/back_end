# 📋 Endpoint de Validación de Archivos Excel

## Descripción

El endpoint `/validate-excel-files` valida archivos Excel de Cliente y NIQ según reglas de negocio específicas antes de procesarlos.

## 🎯 Propósito

- **Input**: Dos archivos Excel (cliente y NIQ)
- **Output**: Diccionario con resultado de validación
- **Objetivo**: Verificar que los archivos cumplan con las especificaciones de estructura y contenido

## 📊 Especificaciones de Validación

### Cliente (check_excel_cliente)

**Columnas String Válidas:**
- **Size 6**: `CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | FACT`
- **Size 7**: `CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | FACT`
- **Size 8**: `CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | BASEPACK | FACT`

**Validaciones:**
1. Número de columnas string debe ser 6, 7 u 8
2. Calcula valores únicos por columna específica
3. Cuenta columnas numéricas

### NIQ (check_excel_niq)

**Columnas String Válidas:**
- **Size 5**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | FACT`
- **Size 6**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FACT`
- **Size 7**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | FACT`
- **Size 9**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | PACK | SIZE | FACT`

**Validaciones:**
1. Acepta cualquier número de columnas string
2. Calcula valores únicos por columna específica según tamaño
3. Cuenta columnas numéricas

## 🔍 Reglas de Compatibilidad

El endpoint maneja la validación directamente:

1. **Ejecuta** `check_excel_cliente(df_cliente)` → `[size, mensaje, error, tamanios]`
2. **Ejecuta** `check_excel_niq(df_niq)` → `[size, mensaje, error, tamanios]`
3. **Verifica** si `error == true` en cualquiera → Retorna error
4. **Aplica regla**: `niq_numeric_cols >= cliente_numeric_cols`
5. **Retorna** objeto `tamanios` si todo está correcto

## 📝 Formato de Respuesta

### ✅ Validación Exitosa
```json
{
  "success": true,
  "error": false,
  "message": "Validación exitosa",
  "cliente_tamanios": [3, 4, 5, 2, 3],  // [category_size, market_size, brand_size, product_size?, numeric_cols]
  "niq_tamanios": [3, 4, 5, 2, 4],      // [category_size, market_size, brand_size, product_size?, numeric_cols]
  "cliente_result": [6, "File written successfully", false, [3, 4, 5, 3]],
  "niq_result": [6, "File written successfully", false, [3, 4, 5, 4]],
  "cliente_numeric_cols": 3,
  "niq_numeric_cols": 4,
  "cliente_filename": "datos_cliente.xlsx",
  "niq_filename": "datos_niq.xlsx",
  "cliente_shape": [1000, 9],
  "niq_shape": [5000, 10],
  "validation_timestamp": "2025-01-01T12:00:00"
}
```

### ❌ Error de Validación
```json
{
  "success": false,
  "error": true,
  "message": "Error with the columns of client info",
  "cliente_result": [5, "Error with the columns of client info", true, []],
  "niq_result": [6, "File written successfully", false, [3, 4, 5, 4]],
  "cliente_filename": "datos_cliente.xlsx",
  "niq_filename": "datos_niq.xlsx",
  "cliente_shape": [1000, 8],
  "niq_shape": [5000, 10],
  "validation_timestamp": "2025-01-01T12:00:00"
}
```

### ❌ Error de Compatibilidad Numérica
```json
{
  "success": false,
  "error": true,
  "message": "Error de datos: NIQ tiene 2 columnas numéricas, pero cliente tiene 3. NIQ debe tener >= columnas numéricas que cliente.",
  "cliente_result": [6, "File written successfully", false, [3, 4, 5, 3]],
  "niq_result": [6, "File written successfully", false, [3, 4, 5, 2]],
  "cliente_numeric_cols": 3,
  "niq_numeric_cols": 2,
  "cliente_filename": "datos_cliente.xlsx",
  "niq_filename": "datos_niq.xlsx",
  "cliente_shape": [1000, 9],
  "niq_shape": [5000, 8],
  "validation_timestamp": "2025-01-01T12:00:00"
}
```

## 🚀 Uso del Endpoint

### cURL
```bash
curl -X POST "http://localhost:8000/validate-excel-files" \
  -H "Content-Type: multipart/form-data" \
  -F "cliente_file=@datos_cliente.xlsx" \
  -F "niq_file=@datos_niq.xlsx"
```

### JavaScript/React
```javascript
const validateFiles = async (clienteFile, niqFile) => {
  const formData = new FormData();
  formData.append('cliente_file', clienteFile);
  formData.append('niq_file', niqFile);

  try {
    const response = await fetch('http://localhost:8000/validate-excel-files', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();
    
    if (result.success) {
      console.log('✅ Archivos válidos:', result.cliente_tamanios, result.niq_tamanios);
      return result;
    } else {
      console.error('❌ Error de validación:', result.message);
      throw new Error(result.message);
    }
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};
```

### Python
```python
import requests

def validate_excel_files(cliente_path, niq_path):
    with open(cliente_path, 'rb') as cliente_file, open(niq_path, 'rb') as niq_file:
        files = {
            'cliente_file': cliente_file,
            'niq_file': niq_file
        }
        
        response = requests.post('http://localhost:8000/validate-excel-files', files=files)
        result = response.json()
        
        if result['success']:
            print(f"✅ Validación exitosa")
            print(f"Cliente tamaños: {result['cliente_tamanios']}")
            print(f"NIQ tamaños: {result['niq_tamanios']}")
            return result
        else:
            print(f"❌ Error: {result['message']}")
            return result

# Uso
result = validate_excel_files('cliente.xlsx', 'niq.xlsx')
```

## 🔧 Casos de Uso

1. **Pre-validación**: Validar archivos antes de procesamiento completo
2. **Control de Calidad**: Verificar estructura de datos
3. **Feedback Inmediato**: Informar errores específicos al usuario
4. **Compatibilidad**: Asegurar que Cliente y NIQ son compatibles

## 📊 Interpretación de Tamaños

### Cliente
- `tamanios[0]`: Número de categorías únicas
- `tamanios[1]`: Número de mercados únicos (CHANNEL)
- `tamanios[2]`: Número de marcas únicas
- `tamanios[3]`: Número de productos únicos (si aplica)
- `tamanios[-1]`: Número de columnas numéricas

### NIQ
- `tamanios[0]`: Número de categorías únicas
- `tamanios[1]`: Número de mercados únicos (MARKETS)
- `tamanios[2]`: Número de marcas únicas
- `tamanios[3]`: Número de productos únicos (si aplica)
- `tamanios[-1]`: Número de columnas numéricas

## ⚠️ Consideraciones

- Los archivos no se almacenan en memoria durante la validación
- La validación es rápida y eficiente
- Retorna información detallada para debugging
- Compatible con todos los formatos Excel (.xlsx, .xls)
