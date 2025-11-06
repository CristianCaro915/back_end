# Endpoint: `/calculate-coverage-channels`

## Descripción
Calcula la cobertura de mercado a nivel de canales (DTT/Traditional y Modern), separando los datos y generando métricas específicas para cada canal.

## Método
`POST`

## URL
```
http://localhost:8000/calculate-coverage-channels
```

## Request Body

```json
{
  "drill_down_level": "Channels"
}
```

### Parámetros

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `drill_down_level` | string | Sí | Nivel de análisis (ej: "Channels") |

## Prerequisitos

Antes de llamar a este endpoint, debes haber ejecutado:

1. `POST /validate-excel-files` - Para cargar los datos
2. `POST /unify-channels-brands` - Para crear las columnas `Homologado_C` que identifican los canales

## Response

### Estructura de Respuesta Exitosa

```json
{
  "DTT": [
    {
      // Scorecard para DTT
      "manufacturer": "PEPSICO",
      "channel": "DTT",
      "time_frame": "Feb 2025",
      "mat_yago": 72.5,
      "latest_mat": 74.3,
      "difference": 1.8,
      "trend_niq": 5.2,
      "trend_pepsico": 3.8,
      "wd": 85.0,
      "nd": 92.0,
      "share": 24.3,
      "drill_down_level": "Channels"
    },
    {
      // Chart data para DTT
      "title": "Market Coverage Trends - DTT",
      "description": "Market trends for DTT channel",
      "data": [
        {
          "period": "Jan 2024",
          "coverage": 72.5,
          "nielseniq": 550000,
          "client": 750000
        },
        {
          "period": "Feb 2024",
          "coverage": 73.1,
          "nielseniq": 560000,
          "client": 765000
        }
        // ... más períodos
      ],
      "metadata": {
        "max_months": 18,
        "date_range": "Jan 2024 - Jun 2025"
      }
    }
  ],
  "Modern": [
    {
      // Scorecard para Modern (misma estructura que DTT)
      "manufacturer": "PEPSICO",
      "channel": "Modern",
      "time_frame": "Feb 2025",
      "mat_yago": 68.2,
      "latest_mat": 70.5,
      "difference": 2.3,
      "trend_niq": 6.1,
      "trend_pepsico": 4.5,
      "wd": 85.0,
      "nd": 92.0,
      "share": 24.3,
      "drill_down_level": "Channels"
    },
    {
      // Chart data para Modern (misma estructura que DTT)
      "title": "Market Coverage Trends - Modern",
      "description": "Market trends for Modern channel",
      "data": [
        // ... data similar a DTT
      ],
      "metadata": {
        "max_months": 18,
        "date_range": "Jan 2024 - Jun 2025"
      }
    }
  ]
}
```

### Estructura de Respuesta con Error

```json
{
  "error": "Descripción del error"
}
```

## Campos de la Scorecard

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `manufacturer` | string | Nombre del fabricante (desde NIQ) |
| `channel` | string | Canal específico ("DTT" o "Modern") |
| `time_frame` | string | Período más reciente analizado |
| `mat_yago` | float | Cobertura MAT hace 1 año |
| `latest_mat` | float | Cobertura MAT más reciente |
| `difference` | float | Diferencia entre `latest_mat` y `mat_yago` |
| `trend_niq` | float | Tendencia NIQ: (último año / año anterior - 1) * 100% |
| `trend_pepsico` | float | Tendencia Cliente: (último año / año anterior - 1) * 100% |
| `wd` | float | Weighted Distribution (99 = fallback) |
| `nd` | float | Numeric Distribution (99 = fallback) |
| `share` | float | Market Share (99 = fallback) |
| `drill_down_level` | string | Nivel de análisis |

## Proceso Interno

### 1. Igualar Columnas Numéricas
- Compara la cantidad de columnas numéricas entre NIQ y Cliente
- Elimina columnas extra del DataFrame más grande
- Si es análisis bimensual, asegura que haya cantidad par de columnas

### 2. Limpiar Datos NIQ
- Elimina filas donde columnas no numéricas sean nulas
- Reporta cantidad de filas eliminadas

### 3. Filtrar por Fact y Canal (NIQ)
- Busca automáticamente la columna que contiene el fact de "Sales"
- Filtra solo filas con fact = Sales
- Agrupa por `Homologado_C` (canal)
- Excluye filas marcadas como 'NO NIQ'

### 4. Ajustar Unidades (Cliente)
Según `input_data_type` configurado:
- **Grams**: Normaliza a gramos
- **Kilograms**: Convierte a kilogramos
- **Liters**: Convierte a litros
- **Milliliters**: Convierte a mililitros

Luego agrupa por `Homologado_C` y excluye 'NO NIQ'

### 5. Aplicar Periodicidad
Si es análisis bimensual:
- Agrupa columnas numéricas por pares
- Suma valores de cada par
- Concatena nombres de columnas (ej: "Jan/Feb")

### 6. Calcular Métricas por Canal

Para **DTT** y **Modern** separadamente:

#### Cobertura (Coverage)
```
Coverage = (NIQ / Cliente) * 100
```
Calculada sobre un período móvil de 1 año (MAT - Moving Annual Total)

#### Tendencias
```
trend_niq = ((suma_ultimo_año / suma_año_anterior) - 1) * 100
trend_client = ((suma_ultimo_año / suma_año_anterior) - 1) * 100
```

Donde:
- **Análisis mensual**: 1 año = 12 períodos
- **Análisis bimensual**: 1 año = 6 períodos

### 7. Construir Respuesta
- Genera scorecard con métricas agregadas
- Genera chart con datos históricos período a período
- Retorna estructura separada para DTT y Modern

## Valores Fallback

Si hay errores en el cálculo:
- `trend_niq`: 99.9
- `trend_pepsico`: 99.9
- `wd`: 99
- `nd`: 99
- `share`: 99

El endpoint continúa funcionando y retorna mensaje en el campo `error` del scorecard correspondiente.

## Errores Comunes

### Error: "No hay datos cargados en el DataManager"
**Solución**: Ejecutar primero `POST /validate-excel-files`

### Error: "No hay fact_mapping configurado"
**Solución**: Ejecutar primero `POST /unify-channels-brands`

### Error: "No existe la columna Homologado_C"
**Solución**: Ejecutar primero `POST /unify-channels-brands`

### Error: "No se encontró la columna con el fact 'Vtas EQ2'"
**Solución**: Verificar que el `fact_mapping` en `unify-channels-brands` tenga la clave 'Sales' correctamente mapeada

### Error: "No hay datos para el canal DTT/Modern"
**Causa**: No hay datos filtrados para ese canal específico
**Verificar**: 
- Que el mapeo de canales en `unify-channels-brands` sea correcto
- Que los datos de entrada tengan información para ambos canales

## Ejemplo de Uso con cURL

```bash
curl -X POST "http://localhost:8000/calculate-coverage-channels" \
  -H "Content-Type: application/json" \
  -d '{
    "drill_down_level": "Channels"
  }'
```

## Ejemplo de Uso con Python

```python
import requests

url = "http://localhost:8000/calculate-coverage-channels"
payload = {
    "drill_down_level": "Channels"
}

response = requests.post(url, json=payload)
data = response.json()

# Acceder a datos de DTT
dtt_scorecard = data["DTT"][0]
dtt_chart = data["DTT"][1]

# Acceder a datos de Modern
modern_scorecard = data["Modern"][0]
modern_chart = data["Modern"][1]

print(f"DTT Coverage: {dtt_scorecard['latest_mat']}%")
print(f"Modern Coverage: {modern_scorecard['latest_mat']}%")
```

## Notas Importantes

1. **Orden de ejecución**: Este endpoint debe ejecutarse DESPUÉS de `validate-excel-files` y `unify-channels-brands`

2. **Canales soportados**: 
   - DTT (Traditional)
   - Modern
   - Se excluyen automáticamente registros marcados como 'NO NIQ'

3. **Periodicidad**: 
   - Se adapta automáticamente según `data_manager.periodicity`
   - Mensual: 12 períodos por año
   - Bimensual: 6 períodos por año

4. **Logging**: Todas las operaciones se registran en el logger para debugging

5. **Manejo de errores**: El endpoint está diseñado para no fallar completamente. Si un canal tiene problemas, el otro aún puede retornar datos válidos.

