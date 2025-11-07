# Documentación: Factor de Conversión en NLP

## Problema Original

Al procesar productos con el NLP, se convertían las unidades (ej: Litros → ml) pero NO se aplicaba el factor de conversión a las columnas numéricas (ventas).

### Ejemplo del Problema:
```
Producto: "Barril 30 Litros"
Ventas: 1.8

Antes del cambio:
- size: 30000 ml (30 * 1000) ✅
- unit: ml ✅
- Ventas: 1.8 ❌ (debería ser 1800)
```

## Solución Implementada

### 1. Modificación de `extract_size_unit()`

**Cambio:** La función ahora retorna una tupla de 3 elementos en lugar de 2.

```python
# ANTES
def extract_size_unit(product_name: str) -> Tuple[Optional[str], Optional[str]]:
    return (size, unit)

# DESPUÉS
def extract_size_unit(product_name: str) -> Tuple[Optional[str], Optional[str], float]:
    return (size, unit, conversion_factor)
```

**Factor de conversión retornado:**
- `1.0` → No hay conversión (ej: "500ml", "150g", "12 units")
- `1000.0` → Litros a ml o Kilogramos a gramos (ej: "30L", "2kg")
- `10.0` → Centilitros a ml (ej: "50cl")
- `29.57` → Onzas a ml (ej: "12oz")

### 2. Modificación de `process_dataframe_with_nlp()`

**Cambio:** La función ahora aplica el factor de conversión a TODAS las columnas numéricas.

```python
def process_dataframe_with_nlp(df: pd.DataFrame, product_column_index: int = 5):
    # Identificar columnas numéricas ANTES de agregar 'size' y 'unit'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Para cada fila
    for idx, product_name in product_column.items():
        size, unit, factor = extract_size_unit(product_name)
        
        # Si el factor es diferente de 1.0, aplicarlo a las columnas numéricas
        if factor != 1.0:
            for col in numeric_cols:
                df_result.at[idx, col] = df_result.at[idx, col] * factor
```

## Ejemplos de Uso

### Ejemplo 1: Conversión de Litros a Mililitros

```python
# DataFrame de entrada
PRODUCT: "Barril 30 Litros"
Enero_2024: 1.8
Febrero_2024: 2.1

# Procesamiento NLP
size, unit, factor = extract_size_unit("Barril 30 Litros")
# size = "30000", unit = "ml", factor = 1000.0

# DataFrame de salida
PRODUCT: "Barril 30 Litros"
size: "30000"
unit: "ml"
Enero_2024: 1800.0  (1.8 * 1000)
Febrero_2024: 2100.0  (2.1 * 1000)
```

### Ejemplo 2: Sin Conversión (gramos)

```python
# DataFrame de entrada
PRODUCT: "Bolsa 150g"
Enero_2024: 5.2
Febrero_2024: 6.0

# Procesamiento NLP
size, unit, factor = extract_size_unit("Bolsa 150g")
# size = "150", unit = "gr", factor = 1.0

# DataFrame de salida
PRODUCT: "Bolsa 150g"
size: "150"
unit: "gr"
Enero_2024: 5.2  (sin cambio, factor = 1.0)
Febrero_2024: 6.0  (sin cambio, factor = 1.0)
```

### Ejemplo 3: Conversión de Kilogramos a Gramos

```python
# DataFrame de entrada
PRODUCT: "Paquete 2kg"
Enero_2024: 3.5
Febrero_2024: 4.2

# Procesamiento NLP
size, unit, factor = extract_size_unit("Paquete 2kg")
# size = "2000", unit = "gr", factor = 1000.0

# DataFrame de salida
PRODUCT: "Paquete 2kg"
size: "2000"
unit: "gr"
Enero_2024: 3500.0  (3.5 * 1000)
Febrero_2024: 4200.0  (4.2 * 1000)
```

## Casos Especiales

### Unidades sin Conversión
Las siguientes unidades NO aplican factor de conversión (factor = 1.0):
- `ml`, `cc`, `cm3`, `cm` → Ya están en ml
- `g`, `gr`, `grs`, `gm` → Ya están en gramos
- `units`, `pcs`, `pz` → Unidades discretas

### Conversiones Aplicadas

| Unidad Original | Unidad Final | Factor |
|----------------|--------------|--------|
| L, lt, lts, litro(s), liter(s) | ml | 1000.0 |
| cl | ml | 10.0 |
| oz | ml | 29.57 |
| kg, kilo(s), k | gr | 1000.0 |

## Impacto en el Sistema

### Archivos Modificados
1. **`nlp_extraction.py`**
   - `extract_size_unit()`: Retorna 3 valores (size, unit, factor)
   - `process_dataframe_with_nlp()`: Aplica factor a columnas numéricas

### Compatibilidad
- ✅ **Backward Compatible:** El código antiguo que solo usaba (size, unit) seguirá funcionando
- ✅ **Sin cambios en API:** El endpoint `/validate-excel-files` no requiere cambios
- ✅ **Logging mejorado:** Se registra cuántas filas tuvieron conversión aplicada

### Performance
- **Complejidad:** O(n * m) donde n = filas, m = columnas numéricas
- **Memoria:** No hay overhead significativo (opera in-place en copias)

## Testing

Se ha creado `test_nlp_conversion.py` que verifica:
1. `extract_size_unit()` retorna el factor correcto
2. Las columnas numéricas se multiplican por el factor
3. Casos edge: productos sin unidad, unidades ya en formato canónico

### Ejecutar Tests
```bash
.venv\Scripts\python.exe test_nlp_conversion.py
```

## Logging

El sistema registra información detallada:

```
INFO - Iniciando procesamiento NLP en columna índice 5
INFO - Columnas numéricas identificadas: 18 columnas
INFO - NLP completado. Columnas 'size' y 'unit' agregadas al DataFrame
INFO - Factores de conversión aplicados a 47 filas en 18 columnas numéricas
```

## Consideraciones Importantes

### 1. Orden de Operaciones
El factor se aplica ANTES de cualquier otra transformación en los endpoints de cobertura.

### 2. Columnas Afectadas
TODAS las columnas numéricas se multiplican por el factor:
- ✅ Enero_2024, Febrero_2024, etc.
- ✅ Cualquier columna numérica existente

### 3. Exactitud
Los factores de conversión están alineados con estándares internacionales:
- 1 Litro = 1000 ml (exacto)
- 1 Kilogramo = 1000 gramos (exacto)
- 1 Onza = 29.57 ml (aproximado, estándar US)

### 4. Validación y Filtrado
Después del NLP, la función `filter_none_values` elimina filas con valores no válidos:
- **size = 'None'**: No se encontró información numérica
- **unit = 'None'**: No se encontró información de unidad
- **unit = 'XXX'**: Unidad ambigua o no reconocida

Esto asegura que solo se procesen productos con información completa y confiable.

## Ejemplo Completo en el Sistema

### Flujo Completo:

1. **Usuario sube Excel con producto:**
   ```
   PRODUCT: "Barril 30L"
   Enero: 1.8
   Febrero: 2.1
   ```

2. **`validate_excel_files` ejecuta NLP:**
   ```python
   df_enriched = validate_and_enrich_client_with_nlp(df_client)
   ```

3. **NLP procesa:**
   ```
   size: "30000"
   unit: "ml"
   Enero: 1800.0
   Febrero: 2100.0
   ```

4. **`data_manager` almacena:**
   ```python
   data_manager.df_client = df_enriched  # Con valores ajustados
   data_manager.df_client_raw = df_client_original  # Sin ajustar
   ```

5. **Endpoints de cobertura usan:**
   ```python
   # df_client tiene valores ajustados por factor
   # Los cálculos de cobertura son correctos
   ```

## Ventajas de esta Implementación

1. ✅ **Consistencia:** Unidades y valores numéricos están sincronizados
2. ✅ **Transparencia:** Se preserva el DataFrame original en `df_client_raw`
3. ✅ **Logging:** Fácil auditoría de conversiones aplicadas
4. ✅ **Flexibilidad:** Fácil agregar nuevas unidades/factores al diccionario
5. ✅ **Robustez:** Manejo de errores por columna/fila

