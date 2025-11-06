"""
Funciones de validación para archivos Excel de Cliente y NIQ
"""

import pandas as pd
import numpy as np
from typing import List, Any, Tuple
import logging

# Importar el módulo NLP
from nlp_extraction import process_dataframe_with_nlp, filter_none_values

logger = logging.getLogger(__name__)

def check_excel_cliente(df_cliente: pd.DataFrame) -> List[Any]:
    """
    Verificar información del cliente según especificaciones
    
    Returns:
        Lista con [size, mensaje, error, tamanios]
    """
    error = False
    mensaje = "File written successfully"
    tamanios = []
    
    # Contar columnas string (no numéricas)
    string_columns = df_cliente.select_dtypes(exclude=[np.number]).columns
    size = len(string_columns)
    
    # Validar que el tamaño sea 6, 7 u 8
    if size not in [6, 7, 8]:
        error = True
        mensaje = "Error with the columns of client info"
        return [size, mensaje, error, tamanios]
    
    # Procesar según el tamaño
    if size == 6:
        # CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | FACT
        size_category = df_cliente.iloc[:, 0].nunique()  # Column 1 (index 0)
        size_market = df_cliente.iloc[:, 3].nunique()    # Column 4 (index 3) - CHANNEL
        size_brand = df_cliente.iloc[:, 4].nunique()     # Column 5 (index 4) - BRAND
        tamanios.extend([size_category, size_market, size_brand])
        
    elif size == 7:
        # CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | FACT
        size_category = df_cliente.iloc[:, 0].nunique()  # Column 1 (index 0)
        size_market = df_cliente.iloc[:, 3].nunique()    # Column 4 (index 3) - CHANNEL
        size_brand = df_cliente.iloc[:, 4].nunique()     # Column 5 (index 4) - BRAND
        size_product = df_cliente.iloc[:, 5].nunique()   # Column 6 (index 5) - SKU
        tamanios.extend([size_category, size_market, size_brand, size_product])
        
    elif size == 8:
        # CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | BASEPACK | FACT
        size_category = df_cliente.iloc[:, 0].nunique()  # Column 1 (index 0)
        size_market = df_cliente.iloc[:, 3].nunique()    # Column 4 (index 3) - CHANNEL
        size_brand = df_cliente.iloc[:, 4].nunique()     # Column 5 (index 4) - BRAND
        size_product = df_cliente.iloc[:, 5].nunique()   # Column 6 (index 5) - SKU
        tamanios.extend([size_category, size_market, size_brand, size_product])
    
    # Contar columnas numéricas (incluir datetime por si algunas columnas son fechas)
    numeric_columns = df_cliente.select_dtypes(include=[np.number, 'datetime64']).columns
    num_numeric_cols = len(numeric_columns)
    tamanios.append(num_numeric_cols)
    
    return [size, mensaje, error, tamanios]


def check_excel_niq(df_niq: pd.DataFrame) -> List[Any]:
    """
    Verificar información de NIQ según especificaciones
    
    Returns:
        Lista con [size, mensaje, error, tamanios]
    """
    error = False
    mensaje = "File written successfully"
    tamanios = []
    
    # Contar columnas string (no numéricas)
    string_columns = df_niq.select_dtypes(exclude=[np.number]).columns
    size = len(string_columns)
    
    # Procesar según el tamaño
    if size == 5:
        # MANUFACTURER | MARKETS | CATEGORY | BRAND | FACT
        size_category = df_niq.iloc[:, 2].nunique()  # Column 3 (index 2) - CATEGORY
        size_market = df_niq.iloc[:, 1].nunique()    # Column 2 (index 1) - MARKETS
        size_brand = df_niq.iloc[:, 3].nunique()     # Column 4 (index 3) - BRAND
        tamanios.extend([size_category, size_market, size_brand])
        
    elif size in [6, 7, 9]:
        # 6: MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FACT
        # 7: MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | FACT
        # 9: MANUFACTURER | MARKETS | CATEGORY | BRAND | SKU | FLAVOR | PACK | SIZE | FACT
        size_category = df_niq.iloc[:, 2].nunique()  # Column 3 (index 2) - CATEGORY
        size_market = df_niq.iloc[:, 1].nunique()    # Column 2 (index 1) - MARKETS
        size_brand = df_niq.iloc[:, 3].nunique()     # Column 4 (index 3) - BRAND
        size_product = df_niq.iloc[:, 4].nunique()   # Column 5 (index 4) - SKU
        tamanios.extend([size_category, size_market, size_brand, size_product])
    
    # Contar columnas numéricas (incluir datetime por si algunas columnas son fechas)
    numeric_columns = df_niq.select_dtypes(include=[np.number, 'datetime64']).columns
    num_numeric_cols = len(numeric_columns)
    tamanios.append(num_numeric_cols)
    
    return [size, mensaje, error, tamanios]


def validate_client_with_nlp(df_client: pd.DataFrame, non_num_cols: int) -> Tuple[pd.DataFrame, str, bool, int]:
    """
    Valida y enriquece el DataFrame de cliente con información de size y unit extraída mediante NLP.
    Esta función se ejecuta solo cuando el cliente tiene 7 u 8 columnas no numéricas.
    
    Args:
        df_client: DataFrame con información del cliente
        non_num_cols: Número de columnas no numéricas
    
    Returns:
        Tuple[pd.DataFrame, str, bool, int]: (DataFrame enriquecido, mensaje, error, filas_eliminadas)
    """
    try:
        logger.info(f"=== Iniciando validación NLP para cliente con {non_num_cols} columnas no numéricas ===")
        
        # Validar que el DataFrame tenga suficientes columnas
        if non_num_cols not in [7, 8]:
            return df_client, "NLP no aplicable: solo se usa con 7 u 8 columnas no numéricas", False, 0
        
        # La columna de productos (SKU) está en la posición 5 (índice 5)
        # Estructura: CATEGORY | SEGMENT | COUNTRY | CHANNEL | BRAND | SKU | ...
        product_column_index = 5
        
        logger.info(f"Aplicando NLP a columna de productos (índice {product_column_index})")
        
        # Aplicar NLP para extraer size y unit
        df_enriched = process_dataframe_with_nlp(df_client, product_column_index=product_column_index)
        
        # Filtrar filas donde size o unit sean 'None'
        df_filtered, rows_deleted = filter_none_values(df_enriched)
        
        if rows_deleted > 0:
            logger.warning(f"Se eliminaron {rows_deleted} filas donde size o unit eran 'None'")
            mensaje = f"NLP aplicado exitosamente. Se eliminaron {rows_deleted} filas con valores 'None' en size/unit"
        else:
            logger.info("NLP aplicado exitosamente. No se eliminaron filas")
            mensaje = "NLP aplicado exitosamente. Todas las filas tienen size y unit válidos"
        
        logger.info(f"DataFrame resultante: {len(df_filtered)} filas, {len(df_filtered.columns)} columnas")
        logger.info(f"Columnas agregadas: 'size' y 'unit'")
        
        return df_filtered, mensaje, False, rows_deleted
        
    except Exception as e:
        error_msg = f"Error aplicando NLP: {str(e)}"
        logger.error(error_msg)
        return df_client, error_msg, True, 0
