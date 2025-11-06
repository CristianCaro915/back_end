"""
Módulo de extracción NLP de size y unit desde nombres de productos.
Basado en el notebook NLP_v2.ipynb
"""

import re
import pandas as pd
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# Configuración y normalización
# -----------------------------

# Diccionario de unidades con conversiones
ALIAS_TO_CANONICAL = {
    # Volumen directo a ml
    "ml": ("ml", 1),
    "cc": ("ml", 1), "cm3": ("ml", 1), "cm": ("ml", 1),
    "cl": ("ml", 10),
    "l": ("ml", 1000), "lt": ("ml", 1000), "lts": ("ml", 1000),
    "litro": ("ml", 1000), "litros": ("ml", 1000),
    "milliliter": ("ml", 1), "milliliters": ("ml", 1),
    "millilitre": ("ml", 1), "millilitres": ("ml", 1),
    "mililiter": ("ml", 1), "mililiters": ("ml", 1),
    "millilitro": ("ml", 1), "millilitros": ("ml", 1),
    "mililitro": ("ml", 1), "mililitros": ("ml", 1),
    "liter": ("ml", 1000), "liters": ("ml", 1000),
    "litre": ("ml", 1000), "litres": ("ml", 1000),

    # Masa directo a gr
    "g": ("gr", 1), "gr": ("gr", 1), "grs": ("gr", 1),
    "gm": ("gr", 1), "gms": ("gr", 1),
    "gram": ("gr", 1), "grams": ("gr", 1),
    "gramo": ("gr", 1), "gramos": ("gr", 1),
    "kg": ("gr", 1000), "kgs": ("gr", 1000),
    "kilo": ("gr", 1000), "kilos": ("gr", 1000),
    "k": ("gr", 1000),

    # Otras
    "oz": ("ml", 29.57),
}

# Lista de tamaños volumétricos típicos (ml)
COMMON_VOLUME_SIZES = {
    187, 200, 237, 250, 270, 275, 300, 310, 312, 320, 330, 340, 350, 355, 375,
    400, 410, 420, 440, 450, 473, 480, 500, 550, 568, 590, 600, 620, 650, 660,
    680, 700, 710, 720, 740, 750, 770, 800, 850, 900, 940, 950, 970, 990, 1000,
    1125, 1180, 1200, 1250, 1500, 1750, 2000, 2250, 2500, 2700, 3000, 3500, 4000, 5000
}

# Palabras genéricas de contenedor
CONTAINER_HINTS = {
    "can", "tin", "lata", "bottle", "botella", "bouteille", "flasche", "bottiglia",
    "jar", "frasco", "pouch", "sachet", "pack", "box", "caja", "estuche", "carton",
    "display", "bundle", "tray", "keg", "brick", "tetra", "tetrapak", "tetra-pack",
    "tube", "tubo", "ampoule", "ampolla", "vial", "bag", "bolsa"
}


def _canon_from_alias(unit_text: str):
    """Obtiene unidad canónica y factor de conversión desde alias"""
    u = unit_text.strip().lower().replace(' ', '').rstrip('.')
    return ALIAS_TO_CANONICAL.get(u, None)


def _num_to_str(val: Union[int, float]) -> str:
    """Render numérico estable: enteros sin .0, decimales sin ceros de cola."""
    if isinstance(val, int):
        return str(val)
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-6:
            return str(int(round(f)))
        s = f"{f:.6f}".rstrip('0').rstrip('.')
        return s
    except Exception:
        return str(val)


def _to_canonical(num_str: str, unit_str: str) -> Tuple[Union[int, float], str]:
    """Convierte número+unidad a (valor, unidad_canónica) con conversiones."""
    num = float(num_str.replace(',', '.'))
    m = _canon_from_alias(unit_str)
    if not m:
        return num, "XXX"
    canonical_unit, factor = m
    val = num * factor
    if abs(val - round(val)) < 1e-6:
        val = int(round(val))
    return val, canonical_unit


def _has_container_hint(text_lower: str) -> bool:
    """True si hay palabras de envase/packaging."""
    words = set(re.findall(r'[a-zA-Záéíóúñüäöëïç\-]+', text_lower))
    words = {w.lower() for w in words}
    return len(words.intersection(CONTAINER_HINTS)) > 0


def _is_common_volume(n_str: str) -> bool:
    """True si el número está en la lista de volúmenes comunes"""
    try:
        f = float(n_str.replace(',', '.'))
    except Exception:
        return False
    if abs(f - round(f)) < 1e-6:
        return int(round(f)) in COMMON_VOLUME_SIZES
    return False


def _looks_like_volume(n_str: str) -> bool:
    """Heurística: ¿parece volumen en ml?"""
    try:
        f = float(n_str.replace(',', '.'))
    except Exception:
        return False
    if abs(f - round(f)) < 1e-6:
        ni = int(round(f))
        if ni in COMMON_VOLUME_SIZES:
            return True
        return 180 <= ni <= 5000 and (ni % 100 == 0 or ni % 50 == 0)
    return 5.0 <= f <= 5000.0


def _is_plausible_packcount(n_str: str) -> bool:
    """Conteos de pack típicos"""
    try:
        n = int(float(n_str.replace(',', '.')))
    except Exception:
        return False
    return 2 <= n <= 72


# -----------------------------
# Regex Patterns
# -----------------------------

_UNIT_ALIASES = sorted(ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
UNIT_RE = r'(?:' + '|'.join(re.escape(u) for u in _UNIT_ALIASES) + r')'
NUM_RE = r'(\d+(?:[.,]\d+)?)'
NUM_RE_NOGRP = r'\d+(?:[.,]\d+)?'
UNIT_FOLLOW = r'(?=$|[^a-zA-Z]|[xX×/])'

# Patrones
PAT_NUM_UNIT_SEP = re.compile(rf'{NUM_RE}\s*({UNIT_RE}){UNIT_FOLLOW}', re.I)
PAT_NUM_UNIT_FUSED = re.compile(rf'{NUM_RE}({UNIT_RE}){UNIT_FOLLOW}', re.I)
PAT_PACK = re.compile(rf'(\d+)\s*[*xX×/]\s*({NUM_RE_NOGRP})(?:\s*({UNIT_RE}){UNIT_FOLLOW})?', re.I)

UNITS_TOK = r'(?:u|un|ud|uds|und\.?|und|unid(?:ad(?:es)?)?|units?|pcs?|pz|pieza(?:s)?)'
PAT_UNITS_BOTH = re.compile(rf'(\d+)\s*{UNITS_TOK}\s*/\s*(\d+)\s*{UNITS_TOK}\b', re.I)
PAT_UNITS_FIRST = re.compile(rf'(\d+)\s*{UNITS_TOK}\s*/\s*(\d+)\b', re.I)
PAT_SLASH_CAJA = re.compile(r'(\d+)\s*/\s*(?:caja|cj|c\/j|box|case)\b', re.I)
PAT_WORD_NUM = re.compile(r'(?:estuche(?:s)?|pack|paquete(?:s)?|caja(?:s)?|box(?:es)?|blister(?:s)?|display(?:s)?|case(?:s)?)\D*(\d+)\b', re.I)
PAT_ANY_NUM = re.compile(rf'\b{NUM_RE_NOGRP}\b')


def extract_size_unit(product_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrae (size, unit) de un nombre de producto.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (size, unit) donde:
            - unit puede ser: 'ml', 'gr', 'units', 'XXX', o None
            - size es string que representa el valor numérico
    """
    if not product_name or not isinstance(product_name, str):
        return None, None

    s = product_name.strip()
    s_clean = s.lower()
    # Normalizaciones
    s_clean = (s_clean
               .replace('c.c.', 'cc')
               .replace('c.c', 'cc')
               .replace('Â°', '°')
               .replace('â°', '°'))

    # 1) "número + unidad" (prioritario)
    m = PAT_NUM_UNIT_SEP.search(s_clean) or PAT_NUM_UNIT_FUSED.search(s_clean)
    if m:
        num, unit = m.group(1), m.group(2)
        val, canonical = _to_canonical(num, unit)
        return (_num_to_str(val), canonical)

    # 2) Paquetes "A/B" o "AxB"
    mp = PAT_PACK.search(s_clean)
    if mp:
        left, right, unit = mp.groups()
        if unit:
            val, canonical = _to_canonical(right, unit)
            return (_num_to_str(val), canonical)
        else:
            # Sin unidad explícita: priorizar COMMON_VOLUME_SIZES
            left_common = _is_common_volume(left)
            right_common = _is_common_volume(right)
            if right_common and not left_common:
                return (_num_to_str(float(right.replace(',', '.'))), 'ml')
            if left_common and not right_common:
                return (_num_to_str(float(left.replace(',', '.'))), 'ml')
            if left_common and right_common:
                return (_num_to_str(float(right.replace(',', '.'))), 'ml')

            # Heurística volumen vs. pack
            left_vol = _looks_like_volume(left)
            right_vol = _looks_like_volume(right)
            left_pack = _is_plausible_packcount(left)
            right_pack = _is_plausible_packcount(right)

            if right_vol and left_pack and not left_vol:
                return (_num_to_str(float(right.replace(',', '.'))), 'ml')
            if left_vol and right_pack and not right_vol:
                return (_num_to_str(float(left.replace(',', '.'))), 'ml')

            # Señal adicional con contenedor
            if _has_container_hint(s_clean):
                if right_vol:
                    return (_num_to_str(float(right.replace(',', '.'))), 'ml')
                if left_vol:
                    return (_num_to_str(float(left.replace(',', '.'))), 'ml')

    # 3) Conteos de unidades
    mu_both = PAT_UNITS_BOTH.search(s_clean)
    if mu_both:
        n1, n2 = mu_both.group(1), mu_both.group(2)
        return (f"{int(n1)}", "units")

    mu_first = PAT_UNITS_FIRST.search(s_clean)
    if mu_first:
        n1 = mu_first.group(1)
        return (str(int(n1)), "units")

    mcj = PAT_SLASH_CAJA.search(s_clean)
    if mcj:
        return (str(int(mcj.group(1))), "units")

    mpw = PAT_WORD_NUM.search(s_clean)
    if mpw:
        return (str(int(mpw.group(1))), "units")

    # 4) Números sueltos
    anyn = PAT_ANY_NUM.search(s_clean)
    if anyn:
        raw = anyn.group(0)
        if _is_common_volume(raw) or (_has_container_hint(s_clean) and _looks_like_volume(raw)):
            return (_num_to_str(float(raw.replace(',', '.'))), 'ml')
        return (_num_to_str(float(raw.replace(',', '.'))), "XXX")

    return None, None


def process_dataframe_with_nlp(df: pd.DataFrame, product_column_index: int = 5) -> pd.DataFrame:
    """
    Procesa un DataFrame aplicando NLP para extraer size y unit de la columna de productos.
    
    Args:
        df: DataFrame de cliente
        product_column_index: Índice de la columna que contiene los nombres de productos (default: 5)
    
    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas 'size' y 'unit' agregadas
    """
    logger.info(f"Iniciando procesamiento NLP en columna índice {product_column_index}")
    
    if product_column_index >= len(df.columns):
        raise ValueError(f"Índice de columna {product_column_index} fuera de rango. DataFrame tiene {len(df.columns)} columnas.")
    
    # Obtener la columna de productos
    product_column = df.iloc[:, product_column_index]
    
    # Aplicar extracción NLP
    extracted = product_column.apply(extract_size_unit)
    
    # Crear las columnas size y unit
    df['size'] = extracted.apply(lambda x: x[0] if x[0] is not None else 'None')
    df['unit'] = extracted.apply(lambda x: x[1] if x[1] is not None else 'None')
    
    logger.info(f"NLP completado. Columnas 'size' y 'unit' agregadas al DataFrame")
    
    return df


def filter_none_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Filtra filas donde 'size' o 'unit' contengan el texto 'None'.
    
    Args:
        df: DataFrame con columnas 'size' y 'unit'
    
    Returns:
        Tuple[pd.DataFrame, int]: (DataFrame filtrado, cantidad de filas eliminadas)
    """
    initial_rows = len(df)
    
    # Filtrar filas donde size o unit son 'None' (como string)
    df_filtered = df[
        (df['size'] != 'None') & 
        (df['unit'] != 'None')
    ].copy()
    
    rows_deleted = initial_rows - len(df_filtered)
    
    logger.info(f"Filtrado completado. Se eliminaron {rows_deleted} filas de {initial_rows} totales")
    
    return df_filtered, rows_deleted

