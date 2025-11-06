# Endpoint: `/calculate-coverage-brands`

## Descripción General

Endpoint POST que calcula la cobertura de mercado a nivel de marcas específicas. Permite analizar múltiples marcas simultáneamente y retorna métricas detalladas para cada una.

## Método HTTP

`POST`

## URL

```
/calculate-coverage-brands
```

## Request Body

El endpoint acepta un JSON con la siguiente estructura:

```json
{
  "drill_down_level": "Brand",
  "brand_names": ["Brand1", "Brand2", "Brand3"]
}
```

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción | Ejemplo |
|-----------|------|-------------|-------------|---------|
| `drill_down_level` | string | Sí | Nivel de análisis para el cálculo | "Brand" |
| `brand_names` | List[string] | Sí | Lista de nombres de marcas a analizar | ["Doritos", "Cheetos", "Lays"] |

### Modelo Pydantic

```python
class CoverageBrandsRequest(BaseModel):
    drill_down_level: str = Field(
        ...,
        description="Nivel de análisis para el cálculo de cobertura por marcas",
        example="Brand"
    )
    brand_names: List[str] = Field(
        ...,
        description="Lista de nombres de marcas a analizar",
        example=["Brand1", "Brand2", "Brand3"]
    )
```

## Proceso de Cálculo

### Paso 0: Preparación de Datos

- Crea copias locales de:
  - `data_manager.df_client`
  - `data_manager.df_niq`
  - `data_manager.df_client_raw`
  - `data_manager.df_niq_raw`

### Paso 1-2: Alineación y Limpieza

- **Función:** `align_numeric_and_clean()`
- **Operaciones:**
  - Iguala el número de columnas numéricas entre df_client y df_niq
  - Si periodicidad es 'bimonthly', asegura número par de columnas
  - Elimina filas con valores nulos en columnas no numéricas de df_niq

### Paso 3: Filtrado por Fact y Agrupación

- **Función:** `filter_by_fact_and_group()`
- **Operaciones:**
  - Filtra df_niq por fact de ventas (Sales)
  - Agrupa por columna de Brand (posición 3 en NIQ)
  - Ajusta unidades en df_client según `input_data_type` (Grams, Kilograms, Liters, Milliliters)
  - Agrupa df_client por columna 'Homologado_B'
  - Excluye registros marcados como 'NO NIQ'

### Paso 4: Aplicación de Periodicidad

Si `data_manager.periodicity == 'bimonthly'`:
- Agrupa columnas numéricas por pares
- Suma valores de meses consecutivos
- Concatena nombres de columnas (ej: "Enero 21/Febrero 21")

Inicializa variables de métricas:
- `wd = 99` (Weighted Distribution)
- `nd = 99` (Numeric Distribution)
- `share = 99` (Market Share)

### Paso 5: Cálculo de Cobertura y Tendencias por Marca

Para cada marca en `brand_names`:

1. **Normalización:** Convierte nombres a minúsculas y elimina espacios
2. **Filtrado:**
   - NIQ: Filtra df_niq_copy por columna 3 (Brand)
   - Cliente: Filtra df_client_copy por columna 'Homologado_B'
3. **Cálculo de Cobertura:**
   - Formula: `(NIQ / Cliente) * 100`
   - Período: 1 año (12 meses para monthly, 6 para bimonthly)
   - Aproximación: 2 decimales
4. **Cálculo de Tendencias:**
   - `trend_niq`: `((suma_último_año / suma_año_anterior) - 1) * 100`
   - `trend_client`: `((suma_último_año / suma_año_anterior) - 1) * 100`
   - Fallback: `99.9` si el cálculo falla

### Paso 6: Construcción de Respuesta

Para cada marca, genera dos estructuras:

#### 6.1 Scorecard

```json
{
  "manufacturer": "PepsiCo",
  "brand": "Doritos",
  "time_frame": "Feb/Mar 24",
  "mat_yago": 74.25,
  "latest_mat": 76.50,
  "difference": 2.25,
  "trend_niq": 3.1,
  "trend_pepsico": 2.9,
  "wd": 99,
  "nd": 99,
  "share": 99,
  "drill_down_level": "Brand"
}
```

**Campos:**
- `manufacturer`: Nombre del fabricante (extraído de df_niq_copy[0])
- `brand`: Nombre de la marca analizada
- `time_frame`: Período más reciente analizado
- `mat_yago`: Cobertura de hace 1 año (MAT Year Ago)
- `latest_mat`: Cobertura del período más reciente (MAT)
- `difference`: Diferencia entre latest_mat y mat_yago
- `trend_niq`: Tendencia de NIQ (%)
- `trend_pepsico`: Tendencia del cliente (%)
- `wd`: Weighted Distribution (placeholder: 99)
- `nd`: Numeric Distribution (placeholder: 99)
- `share`: Market Share (placeholder: 99)
- `drill_down_level`: Nivel de análisis

#### 6.2 Gráfico de Tendencias

```json
{
  "title": "Doritos Coverage Trends",
  "description": "Doritos trends for the selected period",
  "data": [
    {
      "period": "Abr/May 22",
      "coverage": 72.50,
      "nielseniq": 550000.00,
      "client": 750000.00
    },
    {
      "period": "Jun/Jul 22",
      "coverage": 75.25,
      "nielseniq": 580000.00,
      "client": 780000.00
    }
  ],
  "metadata": {
    "max_months": 18,
    "date_range": "Apr 2022 - Sep 2024"
  }
}
```

**Estructura:**
- `title`: Título del gráfico (incluye nombre de marca)
- `description`: Descripción del análisis
- `data`: Array de períodos con:
  - `period`: Nombre del período
  - `coverage`: Cobertura calculada (%)
  - `nielseniq`: Valor de NIQ
  - `client`: Valor del cliente
- `metadata`: Información adicional sobre el rango de datos

## Response

### Respuesta Exitosa

```json
{
  "Brand1": [
    {
      "manufacturer": "PepsiCo",
      "brand": "Brand1",
      "time_frame": "Feb/Mar 24",
      "mat_yago": 74.25,
      "latest_mat": 76.50,
      "difference": 2.25,
      "trend_niq": 3.1,
      "trend_pepsico": 2.9,
      "wd": 99,
      "nd": 99,
      "share": 99,
      "drill_down_level": "Brand"
    },
    {
      "title": "Brand1 Coverage Trends",
      "description": "Brand1 trends for the selected period",
      "data": [...],
      "metadata": {...}
    }
  ],
  "Brand2": [
    {
      "manufacturer": "PepsiCo",
      "brand": "Brand2",
      ...
    },
    {
      "title": "Brand2 Coverage Trends",
      ...
    }
  ]
}
```

**Estructura:** Diccionario donde cada clave es el nombre de una marca y el valor es un array con:
1. **Índice 0:** Scorecard (diccionario)
2. **Índice 1:** Gráfico de tendencias (diccionario)

### Respuesta con Error

#### Error General
```json
{
  "error": "Mensaje descriptivo del error"
}
```

#### Error por Marca Específica
```json
{
  "Brand1": [
    {"error": "No hay datos para la marca 'Brand1'"},
    {"error": "No hay datos para la marca 'Brand1'"}
  ],
  "Brand2": [
    {...},
    {...}
  ]
}
```

## Validaciones Previas

El endpoint valida:

1. **Datos cargados:** Verifica que `df_client_raw`, `df_niq_raw`, `df_client` y `df_niq` no estén vacíos
2. **Fact mapping:** Verifica que `data_manager.fact_selections` esté configurado
3. **Columna Homologado_B:** Verifica que exista en `df_client`
4. **Marcas proporcionadas:** Verifica que `brand_names` no esté vacío

## Códigos de Estado HTTP

| Código | Descripción |
|--------|-------------|
| 200 | Cálculo exitoso (puede incluir errores parciales por marca) |
| 422 | Error de validación en los parámetros de entrada |
| 500 | Error interno del servidor |

## Dependencias

### Funciones de Utilidad

- `align_numeric_and_clean()` - Alinea columnas numéricas y limpia datos
- `filter_by_fact_and_group()` - Filtra por fact y agrupa datos
- `normalize_string()` - Normaliza strings para comparación (minúsculas, sin espacios)

### DataManager

Requiere los siguientes atributos configurados:
- `df_client` (procesado con NLP)
- `df_niq` (procesado con Homologado_C)
- `df_client_raw` (original)
- `df_niq_raw` (original)
- `fact_selections` (diccionario con mappings de facts)
- `input_data_type` (Value, Grams, Kilograms, Liters, Milliliters)
- `periodicity` (monthly, bimonthly)
- `non_num_niq` (cantidad de columnas no numéricas en NIQ)

## Ejemplo de Uso

### cURL

```bash
curl -X POST "http://localhost:8000/calculate-coverage-brands" \
  -H "Content-Type: application/json" \
  -d '{
    "drill_down_level": "Brand",
    "brand_names": ["Doritos", "Cheetos", "Lays"]
  }'
```

### Python (requests)

```python
import requests

url = "http://localhost:8000/calculate-coverage-brands"
payload = {
    "drill_down_level": "Brand",
    "brand_names": ["Doritos", "Cheetos", "Lays"]
}

response = requests.post(url, json=payload)
print(response.json())
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/calculate-coverage-brands', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    drill_down_level: 'Brand',
    brand_names: ['Doritos', 'Cheetos', 'Lays']
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Notas Importantes

1. **Normalización de Nombres:** Los nombres de marcas se normalizan a minúsculas y sin espacios para la comparación
2. **Datos Faltantes:** Si una marca no tiene datos en NIQ o Cliente, se retorna un error específico para esa marca sin detener el procesamiento de las demás
3. **Cálculo de MAT:** Moving Annual Total se calcula sobre los últimos 12 períodos (monthly) o 6 (bimonthly)
4. **Periodicidad Bimonthly:** Las columnas se agrupan por pares y los nombres se concatenan
5. **Fallback Values:** Si los cálculos de tendencia fallan, se usan valores de fallback (99.9)
6. **Logging:** Todas las operaciones se registran en los logs para debugging

## Orden de Ejecución de Endpoints

Para usar este endpoint correctamente, se debe seguir este orden:

1. `POST /validate-excel-files` - Cargar y validar archivos Excel
2. `POST /unify-channels-brands` - Crear columnas homologadas (Homologado_B)
3. `POST /calculate-coverage-brands` - Calcular cobertura por marcas

## Manejo de Errores

El endpoint implementa manejo de errores en múltiples niveles:

1. **Validación de entrada:** Verifica parámetros obligatorios
2. **Validación de estado:** Verifica que los datos necesarios estén cargados
3. **Errores por marca:** Captura errores individuales sin detener el procesamiento completo
4. **Errores de cálculo:** Usa valores de fallback cuando los cálculos fallan
5. **Logging:** Registra todos los errores para facilitar debugging

## Performance

**Complejidad:** O(n * m) donde:
- n = número de marcas a analizar
- m = número de períodos en los datos

**Memoria:** Requiere suficiente memoria para:
- 4 copias completas de DataFrames (client, niq, client_raw, niq_raw)
- n copias adicionales filtradas por marca

**Recomendaciones:**
- Limitar el número de marcas analizadas simultáneamente si los datos son muy grandes
- Monitorear el uso de memoria cuando se analizan muchas marcas

