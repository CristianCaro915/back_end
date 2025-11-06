"""
Script de prueba para las funciones de validación
"""

import pandas as pd
import numpy as np
from validation_functions import check_excel_cliente, check_excel_niq

def create_test_cliente_df(size: int = 6):
    """Crear DataFrame de prueba para cliente"""
    
    if size == 6:
        # CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | FACT
        data = {
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'SEGMENT': ['Seg1', 'Seg2', 'Seg1', 'Seg2'] * 25,
            'COUNTRY': ['US', 'MX', 'US', 'CA'] * 25,
            'CHANNEL': ['Online', 'Retail', 'Online', 'Wholesale'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Price': np.random.uniform(10.0, 50.0, 100)
        }
    elif size == 7:
        # CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | FACT
        data = {
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'SEGMENT': ['Seg1', 'Seg2', 'Seg1', 'Seg2'] * 25,
            'COUNTRY': ['US', 'MX', 'US', 'CA'] * 25,
            'CHANNEL': ['Online', 'Retail', 'Online', 'Wholesale'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'SKU': ['SKU1', 'SKU2', 'SKU3', 'SKU4'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Price': np.random.uniform(10.0, 50.0, 100)
        }
    elif size == 8:
        # CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | BASEPACK | FACT
        data = {
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'SEGMENT': ['Seg1', 'Seg2', 'Seg1', 'Seg2'] * 25,
            'COUNTRY': ['US', 'MX', 'US', 'CA'] * 25,
            'CHANNEL': ['Online', 'Retail', 'Online', 'Wholesale'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'SKU': ['SKU1', 'SKU2', 'SKU3', 'SKU4'] * 25,
            'BASEPACK': ['BP1', 'BP2', 'BP1', 'BP2'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Price': np.random.uniform(10.0, 50.0, 100)
        }
    else:
        # Tamaño inválido para probar error
        data = {
            'Col1': ['A', 'B'] * 50,
            'Col2': ['C', 'D'] * 50,
            'Col3': ['E', 'F'] * 50,
            'Col4': ['G', 'H'] * 50,
            'Col5': ['I', 'J'] * 50,
            'Sales': np.random.randint(100, 1000, 100)
        }
    
    return pd.DataFrame(data)

def create_test_niq_df(size: int = 6):
    """Crear DataFrame de prueba para NIQ"""
    
    if size == 5:
        # MANUFACTURER | MARKETS | CATEGORY | BRAND | FACT
        data = {
            'MANUFACTURER': ['Mfg1', 'Mfg2', 'Mfg1', 'Mfg3'] * 25,
            'MARKETS': ['US', 'MX', 'US', 'CA'] * 25,
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Revenue': np.random.uniform(1000.0, 5000.0, 100),
            'Units': np.random.randint(50, 500, 100)
        }
    elif size == 6:
        # MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FACT
        data = {
            'MANUFACTURER': ['Mfg1', 'Mfg2', 'Mfg1', 'Mfg3'] * 25,
            'MARKETS': ['US', 'MX', 'US', 'CA'] * 25,
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'SKU': ['SKU1', 'SKU2', 'SKU3', 'SKU4'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Revenue': np.random.uniform(1000.0, 5000.0, 100),
            'Units': np.random.randint(50, 500, 100)
        }
    elif size == 7:
        # MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | FACT
        data = {
            'MANUFACTURER': ['Mfg1', 'Mfg2', 'Mfg1', 'Mfg3'] * 25,
            'MARKETS': ['US', 'MX', 'US', 'CA'] * 25,
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'SKU': ['SKU1', 'SKU2', 'SKU3', 'SKU4'] * 25,
            'FLAVOR': ['Vanilla', 'Chocolate', 'Strawberry', 'Mint'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Revenue': np.random.uniform(1000.0, 5000.0, 100),
            'Units': np.random.randint(50, 500, 100)
        }
    elif size == 9:
        # MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | PACK | SIZE | FACT
        data = {
            'MANUFACTURER': ['Mfg1', 'Mfg2', 'Mfg1', 'Mfg3'] * 25,
            'MARKETS': ['US', 'MX', 'US', 'CA'] * 25,
            'CATEGORY': ['Cat1', 'Cat2', 'Cat1', 'Cat3'] * 25,
            'BRAND': ['Brand1', 'Brand2', 'Brand1', 'Brand3'] * 25,
            'SKU': ['SKU1', 'SKU2', 'SKU3', 'SKU4'] * 25,
            'FLAVOR': ['Vanilla', 'Chocolate', 'Strawberry', 'Mint'] * 25,
            'PACK': ['Small', 'Medium', 'Large', 'XL'] * 25,
            'SIZE': ['100g', '200g', '500g', '1kg'] * 25,
            'FACT': ['Fact1', 'Fact2', 'Fact1', 'Fact2'] * 25,
            'Sales': np.random.randint(100, 1000, 100),
            'Volume': np.random.randint(10, 100, 100),
            'Revenue': np.random.uniform(1000.0, 5000.0, 100),
            'Units': np.random.randint(50, 500, 100)
        }
    
    return pd.DataFrame(data)

def validate_compatibility(cliente_result, niq_result):
    """Función auxiliar para validar compatibilidad (replicando lógica del endpoint)"""
    cliente_size, cliente_mensaje, cliente_error, cliente_tamanios = cliente_result
    niq_size, niq_mensaje, niq_error, niq_tamanios = niq_result
    
    # Verificar errores
    if cliente_error:
        return {"success": False, "message": f"Error en cliente: {cliente_mensaje}"}
    if niq_error:
        return {"success": False, "message": f"Error en NIQ: {niq_mensaje}"}
    
    # Verificar columnas numéricas
    cliente_numeric_cols = cliente_tamanios[-1] if cliente_tamanios else 0
    niq_numeric_cols = niq_tamanios[-1] if niq_tamanios else 0
    
    if niq_numeric_cols < cliente_numeric_cols:
        return {
            "success": False, 
            "message": f"NIQ tiene {niq_numeric_cols} columnas numéricas, cliente tiene {cliente_numeric_cols}"
        }
    
    return {"success": True, "message": "Validación exitosa"}

def test_validation_functions():
    """Probar las funciones de validación"""
    
    print("🧪 Probando funciones de validación...")
    print("=" * 50)
    
    # Prueba 1: Cliente válido (size 6) y NIQ válido (size 6)
    print("\n1. Prueba: Cliente size 6, NIQ size 6")
    cliente_df = create_test_cliente_df(6)
    niq_df = create_test_niq_df(6)
    
    cliente_result = check_excel_cliente(cliente_df)
    niq_result = check_excel_niq(niq_df)
    validation = validate_compatibility(cliente_result, niq_result)
    
    print(f"Cliente result: {cliente_result}")
    print(f"NIQ result: {niq_result}")
    print(f"Validation success: {validation['success']}")
    print(f"Message: {validation['message']}")
    
    # Prueba 2: Cliente válido (size 7) y NIQ válido (size 7)
    print("\n2. Prueba: Cliente size 7, NIQ size 7")
    cliente_df = create_test_cliente_df(7)
    niq_df = create_test_niq_df(7)
    
    cliente_result = check_excel_cliente(cliente_df)
    niq_result = check_excel_niq(niq_df)
    validation = validate_compatibility(cliente_result, niq_result)
    
    print(f"Cliente result: {cliente_result}")
    print(f"NIQ result: {niq_result}")
    print(f"Validation success: {validation['success']}")
    print(f"Message: {validation['message']}")
    
    # Prueba 3: Cliente inválido (size 5)
    print("\n3. Prueba: Cliente size 5 (inválido)")
    cliente_df = create_test_cliente_df(5)
    niq_df = create_test_niq_df(6)
    
    cliente_result = check_excel_cliente(cliente_df)
    niq_result = check_excel_niq(niq_df)
    validation = validate_compatibility(cliente_result, niq_result)
    
    print(f"Cliente result: {cliente_result}")
    print(f"NIQ result: {niq_result}")
    print(f"Validation success: {validation['success']}")
    print(f"Message: {validation['message']}")
    
    # Prueba 4: NIQ con menos columnas numéricas que cliente
    print("\n4. Prueba: NIQ con menos columnas numéricas")
    cliente_df = create_test_cliente_df(6)  # 3 columnas numéricas
    niq_df_limited = create_test_niq_df(6)
    # Eliminar algunas columnas numéricas del NIQ
    niq_df_limited = niq_df_limited.drop(['Revenue', 'Units'], axis=1)  # Dejar solo 2 numéricas
    
    cliente_result = check_excel_cliente(cliente_df)
    niq_result = check_excel_niq(niq_df_limited)
    validation = validate_compatibility(cliente_result, niq_result)
    
    print(f"Cliente result: {cliente_result}")
    print(f"NIQ result: {niq_result}")
    print(f"Validation success: {validation['success']}")
    print(f"Message: {validation['message']}")
    
    print("\n" + "=" * 50)
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    test_validation_functions()
