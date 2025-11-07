"""
Utilidades para procesamiento de datos en el análisis de cobertura.
Funciones auxiliares para alinear, limpiar y transformar DataFrames.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def normalize_string(s) -> str:
    """
    Normaliza un string para comparación: convierte a minúsculas y elimina espacios al inicio/final.
    
    Parámetros
    ----------
    s : cualquier tipo que pueda convertirse a string
    
    Retorna
    -------
    str : String normalizado (minúsculas, sin espacios)
    """
    return str(s).lower().strip()


def _numeric_cols_in_order(df: pd.DataFrame) -> List[str]:
    """Devuelve las columnas numéricas preservando el orden actual."""
    return [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]


def align_numeric_and_clean(
    df_client: pd.DataFrame,
    df_niq: pd.DataFrame,
    periodicity: str,
    niq_num_count: int,
    client_num_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paso 1: Igualar cantidad de columnas numéricas tomando como referencia los conteos de los DF 'raw'.
            - Se fija un target = min(niq_num_count, client_num_count).
            - Si un DF (procesado) tiene más numéricas que el target, se eliminan las numéricas extra
              desde el final (comportamiento equivalente al código actual).
            - Si periodicidad es 'bimonthly', fuerza número par de columnas numéricas (elimina la última si queda impar).

    Paso 2: Revisar formato y ajustar en NIQ:
            - Elimina filas donde alguna columna NO numérica esté nula (equivalente a iterar y filtrar).

    Parámetros
    ----------
    df_client : DataFrame procesado del cliente (con size, units, Homologado_C, Homologado_B, etc.)
    df_niq    : DataFrame NIQ (raw o procesado; tú decides qué pasar en cada endpoint)
    periodicity : str ('monthly' | 'bimonthly', etc.)
    niq_num_count : int, # de columnas numéricas en df_niq_raw
    client_num_count : int, # de columnas numéricas en df_client_raw

    Retorna
    -------
    (df_client_out, df_niq_out) : DataFrames con columnas numéricas ajustadas y df_niq limpio en no numéricas.
    """
    # Copias para no mutar los originales
    df_client_out = df_client.copy()
    df_niq_out = df_niq.copy()
    
    logger.info(f"=== Iniciando align_numeric_and_clean ===")
    logger.info(f"Shapes iniciales - Cliente: {df_client_out.shape}, NIQ: {df_niq_out.shape}")
    logger.info(f"Raw counts - NIQ: {niq_num_count}, Cliente: {client_num_count}")
    logger.info(f"Periodicidad: {periodicity}")

    # ---------- Paso 1: Igualar columnas numéricas ----------
    target_num = min(niq_num_count, client_num_count)
    logger.info(f"Target de columnas numéricas: {target_num}")

    # Cliente: si tiene más numéricas que el target, eliminar las sobrantes (del final de la lista de numéricas)
    client_num_cols = _numeric_cols_in_order(df_client_out)
    extra_client = max(0, len(client_num_cols) - target_num)
    if extra_client > 0:
        cols_to_drop = client_num_cols[-extra_client:]
        df_client_out = df_client_out.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Cliente: Eliminadas {extra_client} columnas numéricas extra")

    # NIQ: si tiene más numéricas que el target, eliminar las sobrantes
    niq_num_cols = _numeric_cols_in_order(df_niq_out)
    extra_niq = max(0, len(niq_num_cols) - target_num)
    if extra_niq > 0:
        cols_to_drop = niq_num_cols[-extra_niq:]
        df_niq_out = df_niq_out.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"NIQ: Eliminadas {extra_niq} columnas numéricas extra")

    # Recalcular numéricas después de recortes
    client_num_cols = _numeric_cols_in_order(df_client_out)
    niq_num_cols = _numeric_cols_in_order(df_niq_out)
    logger.info(f"Después de igualar - Cliente: {len(client_num_cols)} cols numéricas, NIQ: {len(niq_num_cols)} cols numéricas")

    # Periodicidad bimensual: asegurar número par de columnas numéricas en ambos
    if periodicity == "bimonthly":
        if len(client_num_cols) % 2 != 0:
            df_client_out = df_client_out.drop(columns=[client_num_cols[-1]], errors="ignore")
            logger.info(f"Cliente: Eliminada última columna numérica para periodicidad bimensual (impar -> par)")
        if len(niq_num_cols) % 2 != 0:
            df_niq_out = df_niq_out.drop(columns=[niq_num_cols[-1]], errors="ignore")
            logger.info(f"NIQ: Eliminada última columna numérica para periodicidad bimensual (impar -> par)")

    # ---------- Paso 2: Limpiar filas con nulos en columnas NO numéricas (en NIQ) ----------
    # Identificar columnas NO numéricas en el df_niq_out recibido
    non_numeric_cols = [c for c in df_niq_out.columns if not np.issubdtype(df_niq_out[c].dtype, np.number)]
    if non_numeric_cols:
        initial_rows = len(df_niq_out)
        df_niq_out = df_niq_out.dropna(subset=non_numeric_cols)
        rows_deleted = initial_rows - len(df_niq_out)
        if rows_deleted > 0:
            logger.info(f"NIQ: Eliminadas {rows_deleted} filas con valores nulos en columnas no numéricas")
        else:
            logger.info(f"NIQ: No se eliminaron filas (sin nulos en columnas no numéricas)")
    
    logger.info(f"Shapes finales - Cliente: {df_client_out.shape}, NIQ: {df_niq_out.shape}")
    logger.info(f"=== Fin align_numeric_and_clean ===")

    return df_client_out, df_niq_out


def filter_by_fact_and_group(
    df_client: pd.DataFrame,
    df_niq: pd.DataFrame,
    fact_sales_name: str,
    column_index_filter_niq: int,
    column_index_filter_client: int,
    input_data_type: str = "Value",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paso 3 generalizado para:
      - Filtrar df_niq por 'fact' (buscando la columna que contiene fact_sales_name de derecha a izquierda),
        luego excluir 'NO NIQ' en la columna indicada por índice y agrupar/sumar numéricas.
      - Ajustar unidades en df_client (mismas reglas actuales), excluir 'NO NIQ' por índice y agrupar/sumar numéricas.

    Parámetros
    ----------
    df_client : DataFrame procesado del cliente (con size, units, Homologado_C, Homologado_B, etc.)
    df_niq    : DataFrame NIQ (con Homologado_C ya incluido)
    fact_sales_name : str  nombre del fact de ventas a localizar (ej. 'Vtas EQ2')
    column_index_filter_niq : int  índice de la columna por la cual filtrar/agrupar NIQ (ej. Homologado_C o Manufacturer)
    column_index_filter_client : int  índice de la columna por la cual filtrar/agrupar Cliente (ej. Homologado_C o Country)
    input_data_type : str  tipo de dato de entrada ('Value', 'Grams', 'Kilograms', 'Liters', 'Milliliters')

    Retorna
    -------
    (df_client_grouped, df_niq_grouped) : DataFrames filtrados y agrupados

    Errores
    -------
    Lanza ValueError con mensajes claros si:
      - No encuentra la columna de fact en df_niq
      - Alguno de los índices de columna no es válido
      - No hay columnas numéricas para sumar
    """
    logger.info(f"=== Iniciando filter_by_fact_and_group ===")
    logger.info(f"Shapes iniciales - Cliente: {df_client.shape}, NIQ: {df_niq.shape}")
    logger.info(f"Fact sales name: '{fact_sales_name}'")
    logger.info(f"NIQ filter index: {column_index_filter_niq}, Cliente filter index: {column_index_filter_client}")
    logger.info(f"Input data type: {input_data_type}")
    
    # --- Copias de trabajo ---
    dfc = df_client.copy()
    dfn = df_niq.copy()

    # --- Validación de índices de columna ---
    if not (0 <= column_index_filter_niq < dfn.shape[1]):
        raise ValueError(f"column_index_filter_niq fuera de rango: {column_index_filter_niq} (NIQ tiene {dfn.shape[1]} columnas)")
    if not (0 <= column_index_filter_client < dfc.shape[1]):
        raise ValueError(f"column_index_filter_client fuera de rango: {column_index_filter_client} (Cliente tiene {dfc.shape[1]} columnas)")

    # --- 3.A Encontrar columna de FACT en df_niq (de derecha a izquierda) ---
    fact_col_name = None
    for col in reversed(dfn.columns.tolist()):
        # buscamos coincidencia exacta por igualdad
        try:
            # convertimos a string para evitar issues con tipos mixtos
            if pd.Series(dfn[col].astype(str)).eq(str(fact_sales_name)).any():
                fact_col_name = col
                logger.info(f"Columna de fact encontrada: '{fact_col_name}'")
                break
        except Exception as e:
            # si alguna columna da problema al castear, la omitimos
            logger.debug(f"No se pudo procesar columna '{col}': {e}")
            continue

    if fact_col_name is None:
        raise ValueError(f"No se encontró en df_niq una columna que contenga el fact '{fact_sales_name}'.")

    # --- 3.B Filtrar df_niq por fact y por 'NO NIQ' en la columna objetivo, luego agrupar y sumar numéricas ---
    initial_niq_rows = len(dfn)
    dfn = dfn[dfn[fact_col_name].astype(str) == str(fact_sales_name)]
    logger.info(f"NIQ: Filas después de filtrar por fact '{fact_sales_name}': {len(dfn)} (eliminadas {initial_niq_rows - len(dfn)})")
    
    niq_group_col = dfn.columns[column_index_filter_niq]
    logger.info(f"NIQ: Columna para filtrar/agrupar: '{niq_group_col}' (índice {column_index_filter_niq})")
    
    if niq_group_col not in dfn.columns:
        raise ValueError(f"La columna NIQ para filtrar/agrupación no existe: idx {column_index_filter_niq}, nombre '{niq_group_col}'")

    # Excluir 'NO NIQ'
    initial_niq_rows = len(dfn)
    dfn = dfn[dfn[niq_group_col].astype(str) != "NO NIQ"]
    logger.info(f"NIQ: Filas después de excluir 'NO NIQ': {len(dfn)} (eliminadas {initial_niq_rows - len(dfn)})")
    logger.info(f"NIQ: Valores únicos en '{niq_group_col}': {dfn[niq_group_col].unique().tolist()}")
    
    # Agrupar y sumar numéricas
    dfn_grouped = dfn.groupby(niq_group_col, as_index=False).sum(numeric_only=True)
    logger.info(f"NIQ: Después de groupby: {dfn_grouped.shape} (grupos: {len(dfn_grouped)})")

    # Validar que haya columnas numéricas disponibles tras el groupby
    niq_numeric_after = dfn_grouped.select_dtypes(include=[np.number]).columns.tolist()
    if len(niq_numeric_after) == 0:
        raise ValueError("No hay columnas numéricas en df_niq para sumar después de filtrar por fact y canal/categoría.")
    logger.info(f"NIQ: {len(niq_numeric_after)} columnas numéricas después de agrupar")

    # --- 3.C Ajustar unidades en df_client según las mismas reglas actuales ---
    if "unit" in dfc.columns:
        logger.info(f"Cliente: Ajustando unidades según input_data_type='{input_data_type}'")
        numeric_cols_client = dfc.select_dtypes(include=[np.number]).columns

        if input_data_type == "Kilograms":
            mask_gr = dfc["unit"] == "gr"
            if mask_gr.any():
                dfc.loc[mask_gr, numeric_cols_client] = dfc.loc[mask_gr, numeric_cols_client] * 1000
                dfc.loc[mask_gr, "unit"] = "kg"
                logger.info(f"Cliente: Convertidas {mask_gr.sum()} filas de 'gr' a 'kg'")
            
            mask_other = (dfc["unit"] != "kg") & (dfc["unit"] != "units")
            if mask_other.any():
                dfc.loc[mask_other, numeric_cols_client] = dfc.loc[mask_other, numeric_cols_client] * 1000
                dfc.loc[mask_other, "unit"] = "kg"
                logger.info(f"Cliente: Convertidas {mask_other.sum()} filas a 'kg'")
            
            mask_units = dfc["unit"] == "units"
            if mask_units.any():
                dfc.loc[mask_units, numeric_cols_client] = dfc.loc[mask_units, numeric_cols_client] * 1 # factor de conversión ausente
                dfc.loc[mask_units, "unit"] = "kg"
                logger.info(f"Cliente: Ajustadas {mask_units.sum()} filas de 'units' a 'kg'")

        elif input_data_type == "Liters":
            mask_ml = dfc["unit"] == "ml"
            if mask_ml.any():
                dfc.loc[mask_ml, numeric_cols_client] = dfc.loc[mask_ml, numeric_cols_client] * 1000
                dfc.loc[mask_ml, "unit"] = "lt"
                logger.info(f"Cliente: Convertidas {mask_ml.sum()} filas de 'ml' a 'lt'")
            
            mask_other = (dfc["unit"] != "lt") & (dfc["unit"] != "units")
            if mask_other.any():
                dfc.loc[mask_other, numeric_cols_client] = dfc.loc[mask_other, numeric_cols_client] * 1000
                dfc.loc[mask_other, "unit"] = "lt"
                logger.info(f"Cliente: Convertidas {mask_other.sum()} filas a 'lt'")
            
            mask_units = dfc["unit"] == "units"
            if mask_units.any():
                dfc.loc[mask_units, numeric_cols_client] = dfc.loc[mask_units, numeric_cols_client] * 1 # factor de conversión ausente
                dfc.loc[mask_units, "unit"] = "lt"
                logger.info(f"Cliente: Ajustadas {mask_units.sum()} filas de 'units' a 'lt'")

    else:
        logger.info("Cliente: No hay columna 'unit', omitiendo ajuste de unidades")

    # --- 3.D Filtrar df_client por 'NO NIQ' y agrupar/sumar numéricas ---
    client_group_col = dfc.columns[column_index_filter_client]
    logger.info(f"Cliente: Columna para filtrar/agrupar: '{client_group_col}' (índice {column_index_filter_client})")
    
    if client_group_col not in dfc.columns:
        raise ValueError(f"La columna Client para filtrar/agrupación no existe: idx {column_index_filter_client}, nombre '{client_group_col}'")

    initial_client_rows = len(dfc)
    dfc = dfc[dfc[client_group_col].astype(str) != "NO NIQ"]
    logger.info(f"Cliente: Filas después de excluir 'NO NIQ': {len(dfc)} (eliminadas {initial_client_rows - len(dfc)})")
    logger.info(f"Cliente: Valores únicos en '{client_group_col}': {dfc[client_group_col].unique().tolist()}")
    
    dfc_grouped = dfc.groupby(client_group_col, as_index=False).sum(numeric_only=True)
    logger.info(f"Cliente: Después de groupby: {dfc_grouped.shape} (grupos: {len(dfc_grouped)})")
    
    client_numeric_after = dfc_grouped.select_dtypes(include=[np.number]).columns.tolist()
    if len(client_numeric_after) == 0:
        raise ValueError("No hay columnas numéricas en df_client para sumar después de filtrar por canal/categoría.")
    logger.info(f"Cliente: {len(client_numeric_after)} columnas numéricas después de agrupar")

    logger.info(f"Shapes finales - Cliente: {dfc_grouped.shape}, NIQ: {dfn_grouped.shape}")
    logger.info(f"=== Fin filter_by_fact_and_group ===")

    # Retornar dataframes listos 
    return dfc_grouped, dfn_grouped


def extract_metrics_from_niq(
    df_niq_raw_copy: pd.DataFrame,
    drill_down_level: str,
    nd_fact_name: str,
    wd_fact_name: str,
    share_fact_name: str,
    non_num_niq: int
) -> tuple[float, float, float]:
    """
    Extrae las métricas ND, WD y Share de df_niq_raw_copy según el drill down level.
    
    Args:
        df_niq_raw_copy: DataFrame de NIQ sin procesar
        drill_down_level: Nivel de análisis ('Total', 'Channels', 'Brand')
        nd_fact_name: Nombre del fact para Numeric Distribution
        wd_fact_name: Nombre del fact para Weighted Distribution
        share_fact_name: Nombre del fact para Share
        non_num_niq: Cantidad de columnas no numéricas en NIQ
    
    Returns:
        tuple[float, float, float]: (nd, wd, share)
    """
    logger.info(f"=== Iniciando extract_metrics_from_niq ===")
    logger.info(f"Drill down level: {drill_down_level}")
    logger.info(f"ND fact name: {nd_fact_name}")
    logger.info(f"WD fact name: {wd_fact_name}")
    logger.info(f"Share fact name: {share_fact_name}")
    
    # Valores por defecto
    nd, wd, share = 99, 99, 99
    
    try:
        # Solo procesar para nivel 'Total'
        if drill_down_level.lower() != 'total':
            logger.info(f"Drill down level '{drill_down_level}' no es 'Total'. Retornando valores por defecto.")
            return nd, wd, share
        
        # Hacer copia local
        df_copy = df_niq_raw_copy.copy()
        logger.info(f"Copia creada. Shape: {df_copy.shape}")
        
        # Identificar última columna no numérica (FACT)
        last_non_numeric_col_idx = non_num_niq - 1
        if last_non_numeric_col_idx < 0 or last_non_numeric_col_idx >= len(df_copy.columns):
            logger.error(f"Índice de última columna no numérica fuera de rango: {last_non_numeric_col_idx}")
            return nd, wd, share
        
        last_non_numeric_col = df_copy.columns[last_non_numeric_col_idx]
        manufacturer_col = df_copy.columns[0]
        
        logger.info(f"Columna de Manufacturer: {manufacturer_col}")
        logger.info(f"Columna de FACT (última no numérica): {last_non_numeric_col}")
        
        # Verificar si la columna Manufacturer tiene al menos una celda nula/vacía
        null_manufacturer_count = df_copy[manufacturer_col].isna().sum()
        empty_manufacturer_count = (df_copy[manufacturer_col].astype(str).str.strip() == '').sum()
        total_null_empty = null_manufacturer_count + empty_manufacturer_count
        
        logger.info(f"Celdas nulas en Manufacturer: {null_manufacturer_count}")
        logger.info(f"Celdas vacías en Manufacturer: {empty_manufacturer_count}")
        logger.info(f"Total nulas/vacías: {total_null_empty}")
        
        if total_null_empty == 0:
            logger.info("No hay celdas nulas/vacías en Manufacturer. Retornando valores por defecto.")
            return nd, wd, share
        
        # Filtrar por filas donde Manufacturer es nulo/vacío
        df_filtered = df_copy[
            df_copy[manufacturer_col].isna() | 
            (df_copy[manufacturer_col].astype(str).str.strip() == '')
        ].copy()
        
        logger.info(f"Filas después de filtrar Manufacturer nulo/vacío: {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            logger.warning("No hay filas después de filtrar por Manufacturer nulo/vacío")
            return nd, wd, share
        
        # Filtrar por facts relevantes
        df_filtered = df_filtered[
            (df_filtered[last_non_numeric_col] == nd_fact_name) |
            (df_filtered[last_non_numeric_col] == wd_fact_name) |
            (df_filtered[last_non_numeric_col] == share_fact_name)
        ].copy()
        
        logger.info(f"Filas después de filtrar por facts: {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            logger.warning("No hay filas después de filtrar por facts relevantes")
            return nd, wd, share
        
        # Obtener última columna numérica (último mes)
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            logger.error("No hay columnas numéricas en df_filtered")
            return nd, wd, share
        
        last_numeric_col = numeric_cols[-1]
        logger.info(f"Última columna numérica (último mes): {last_numeric_col}")
        
        # Caso 1: Exactamente 3 filas (una por cada fact)
        if len(df_filtered) == 3:
            logger.info("Exactamente 3 filas encontradas. Extrayendo métricas directamente.")
            
            for idx, row in df_filtered.iterrows():
                fact_value = row[last_non_numeric_col]
                metric_value = row[last_numeric_col]
                
                if fact_value == nd_fact_name:
                    nd = float(metric_value) if pd.notna(metric_value) else 99.99
                    logger.info(f"ND extraído: {nd}")
                elif fact_value == wd_fact_name:
                    wd = float(metric_value) if pd.notna(metric_value) else 99.99
                    logger.info(f"WD extraído: {wd}")
                elif fact_value == share_fact_name:
                    share = float(metric_value) if pd.notna(metric_value) else 99.99
                    logger.info(f"Share extraído: {share}")
        
        # Caso 2: Más de 3 filas, filtrar por cada fact y tomar el mayor
        else:
            logger.info(f"Más de 3 filas encontradas ({len(df_filtered)}). Ordenando y tomando el mayor valor de cada fact.")
            
            # ND
            df_nd = df_filtered[df_filtered[last_non_numeric_col] == nd_fact_name].copy()
            if len(df_nd) > 0:
                df_nd_sorted = df_nd.sort_values(by=last_numeric_col, ascending=False)
                nd = float(df_nd_sorted.iloc[0][last_numeric_col]) if pd.notna(df_nd_sorted.iloc[0][last_numeric_col]) else 99.99
                logger.info(f"ND extraído (mayor): {nd}")
            else:
                logger.warning(f"No se encontraron filas para ND fact: {nd_fact_name}")
            
            # WD
            df_wd = df_filtered[df_filtered[last_non_numeric_col] == wd_fact_name].copy()
            if len(df_wd) > 0:
                df_wd_sorted = df_wd.sort_values(by=last_numeric_col, ascending=False)
                wd = float(df_wd_sorted.iloc[0][last_numeric_col]) if pd.notna(df_wd_sorted.iloc[0][last_numeric_col]) else 99.99
                logger.info(f"WD extraído (mayor): {wd}")
            else:
                logger.warning(f"No se encontraron filas para WD fact: {wd_fact_name}")
            
            # Share
            df_share = df_filtered[df_filtered[last_non_numeric_col] == share_fact_name].copy()
            if len(df_share) > 0:
                df_share_sorted = df_share.sort_values(by=last_numeric_col, ascending=False)
                share = float(df_share_sorted.iloc[0][last_numeric_col]) if pd.notna(df_share_sorted.iloc[0][last_numeric_col]) else 99.99
                logger.info(f"Share extraído (mayor): {share}")
            else:
                logger.warning(f"No se encontraron filas para Share fact: {share_fact_name}")
        
        logger.info(f"=== Fin extract_metrics_from_niq ===")
        logger.info(f"Métricas finales - ND: {nd}, WD: {wd}, Share: {share}")
        
        return nd, wd, share
    
    except Exception as e:
        logger.error(f"Error en extract_metrics_from_niq: {str(e)}")
        logger.error(f"Retornando valores por defecto (99.99)")
        return 99.99, 99.99, 99.99

