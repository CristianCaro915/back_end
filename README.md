# Excel Validation API - FastAPI

API backend desarrollada con FastAPI para validar archivos Excel de Cliente y NIQ según reglas de negocio específicas.

## 🚀 Características

- **Validación de archivos Excel**: Soporta .xlsx y .xls
- **Reglas de negocio específicas**: Valida estructura de Cliente y NIQ
- **Compatibilidad numérica**: Verifica que NIQ tenga suficientes columnas numéricas
- **CORS habilitado**: Listo para frontend React
- **Documentación automática**: Swagger UI incluido

## 📋 Requisitos

- Python 3.8+
- Entorno virtual activado
- Dependencias instaladas

## 🛠️ Instalación

1. **Crear entorno virtual** (si no lo has hecho):
```bash
python -m venv .env
```

2. **Activar entorno virtual**:
```bash
# Windows
.env\Scripts\activate

# Linux/Mac
source .env/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Iniciar el servidor

```bash
python run_server.py
```

O directamente con uvicorn:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### URLs importantes

- **API Base**: http://localhost:8000
- **Documentación Swagger**: http://localhost:8000/docs
- **Documentación Redoc**: http://localhost:8000/redoc

## 📚 Endpoint Disponible

### POST /validate-excel-files

Valida archivos Excel de Cliente y NIQ según especificaciones de negocio.

**Input**: Dos archivos Excel (cliente_file y niq_file)
**Output**: Diccionario con resultado de validación

## 📊 Especificaciones de Validación

### Cliente
**Columnas String Válidas:**
- **Size 6**: `CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | FACT`
- **Size 7**: `CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | FACT`
- **Size 8**: `CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | BASEPACK | FACT`

### NIQ
**Columnas String Válidas:**
- **Size 5**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | FACT`
- **Size 6**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FACT`
- **Size 7**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | FACT`
- **Size 9**: `MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | PACK | SIZE | FACT`

### Reglas de Compatibilidad
1. Cliente debe tener 6, 7 u 8 columnas string
2. NIQ debe tener ≥ columnas numéricas que Cliente
3. Si hay error en cualquiera → Error general

## 📊 Ejemplos de Uso

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

  const response = await fetch('http://localhost:8000/validate-excel-files', {
    method: 'POST',
    body: formData
  });

  return await response.json();
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
        return response.json()
```

## 📝 Formato de Respuesta

### ✅ Validación Exitosa
```json
{
  "success": true,
  "error": false,
  "message": "Validación exitosa",
  "cliente_tamanios": [3, 4, 5, 3],
  "niq_tamanios": [3, 4, 5, 4],
  "cliente_numeric_cols": 3,
  "niq_numeric_cols": 4,
  "cliente_filename": "datos_cliente.xlsx",
  "niq_filename": "datos_niq.xlsx",
  "cliente_shape": [1000, 9],
  "niq_shape": [5000, 10]
}
```

### ❌ Error de Validación
```json
{
  "success": false,
  "error": true,
  "message": "Error with the columns of client info",
  "cliente_result": [5, "Error with the columns of client info", true, []],
  "niq_result": [6, "File written successfully", false, [3, 4, 5, 4]]
}
```

## 🔧 Estructura del Proyecto

```
backend-fastapi/
├── main.py                    # Aplicación principal FastAPI
├── validation_functions.py    # Funciones de validación
├── start.py                   # Script de inicio básico
├── run_server.py              # Script de inicio mejorado
├── test_validation.py         # Pruebas de validación
├── requirements.txt           # Dependencias
├── README.md                  # Documentación
├── VALIDATION_ENDPOINT.md     # Documentación detallada del endpoint
└── TROUBLESHOOTING.md         # Guía de solución de problemas
```

## 🧪 Probar la API

1. **Inicia el servidor**: `python run_server.py`
2. **Ve a la documentación**: http://localhost:8000/docs
3. **Busca el endpoint**: `/validate-excel-files`
4. **Sube dos archivos Excel** y verifica la respuesta

## 📝 Notas

- La API solo valida archivos, no los almacena
- Retorna información detallada para debugging
- Compatible con todos los formatos Excel (.xlsx, .xls)
- Optimizada para validación rápida y eficiente