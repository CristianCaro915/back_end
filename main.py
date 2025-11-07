from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import io
import logging
from typing import Optional, Dict, List
from datetime import datetime

# Importar funciones de validación
from validation_functions import check_excel_cliente, check_excel_niq, validate_client_with_nlp

# Importar funciones de utilidad para procesamiento de datos
from data_processing_utils import align_numeric_and_clean, filter_by_fact_and_group, normalize_string, extract_metrics_from_niq


class UnifyChannelsBrandsRequest(BaseModel):
    """Modelo Pydantic para los parámetros del endpoint unify-channels-brands"""
    dicc_niq_channels: Dict[str, List[str]] = Field(
        ..., 
        description="Diccionario de canales NIQ {'Modern':[market1,market2],'DTT':[market3,market4]}",
        example={"Modern": ["market1", "market2"], "DTT": ["market3", "market4"]}
    )
    dicc_client_channels: Dict[str, List[str]] = Field(
        ..., 
        description="Diccionario de canales Cliente {'Modern':[market5,market6],'DTT':[market7],'NoNIQ':[market8]}",
        example={"Modern": ["market5", "market6"], "DTT": ["market7"], "NoNIQ": ["market8"]}
    )
    dicc_brand_mapping: Dict[str, List[str]] = Field(
        ..., 
        description="Diccionario de mapeo de marcas {Brand1:[Branda,Brandb,Brandc,Brandd]}",
        example={"Brand1": ["Branda", "Brandb", "Brandc", "Brandd"], "Brand2": ["Brande", "Brandf"]}
    )
    list_product_deletion: List[str] = Field(
        ..., 
        description="Lista de productos a eliminar [Product1,Product2,Product3]",
        example=["Product1", "Product2", "Product3"]
    )
    input_data_type: str = Field(
        ..., 
        description="Tipo de dato",
        pattern="^(Value|Grams|Kilograms|Liters|Milliliters)$",
        example="Value"
    )
    analysis_periodicity: str = Field(
        ..., 
        description="Periodicidad del análisis",
        pattern="^(monthly|bimonthly)$",
        example="monthly"
    )
    fact_mapping: Dict[str, str] = Field(
        ...,
        description="Diccionario de mapeo de facts",
        example={
            "Sales": "Vtas EQ2",
            "Share": "Part. de Vtas EQ2 - Product",
            "Numeric Distribution (ND)": "Weighted Distribution (WD)",
            "Weighted Distribution (WD)": "Dist. Pond. Tiendas Handling"
        }
    )


class CoverageTotalRequest(BaseModel):
    """Modelo Pydantic para los parámetros del endpoint calculate-coverage-total"""
    drill_down_level: str = Field(
        ...,
        description="Nivel de análisis para el cálculo de cobertura",
        pattern="^(Total|Channels|Category|Brand|Product|None)$",
        example="Total"
    )


class CoverageChannelsRequest(BaseModel):
    """Modelo Pydantic para los parámetros del endpoint calculate-coverage-channels"""
    drill_down_level: str = Field(
        ...,
        description="Nivel de análisis para el cálculo de cobertura por canales",
        example="Channels"
    )


class CoverageBrandsRequest(BaseModel):
    """Modelo Pydantic para los parámetros del endpoint calculate-coverage-brands"""
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


class DataManager:
    """
    Clase singleton para manejar los datos de cliente y NIQ durante la sesión de la aplicación.
    Mantiene el estado de los DataFrames y configuraciones de análisis.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Inicializar con valores por defecto
            self.df_client: pd.DataFrame = pd.DataFrame()
            self.df_niq: pd.DataFrame = pd.DataFrame()
            self.df_client_raw: pd.DataFrame = pd.DataFrame()
            self.df_niq_raw: pd.DataFrame = pd.DataFrame()
            self.non_num_client: int = 0
            self.non_num_niq: int = 0
            self.drill_down_level: str = "None"
            self.input_data_type: str = "Empty"
            self.periodicity: str = "Empty"
            self.fact_selections: dict = {}
            self._initialized = True
    
    def reset_data(self):
        """Resetear todos los datos a valores por defecto"""
        self.df_client = pd.DataFrame()
        self.df_niq = pd.DataFrame()
        self.df_client_raw = pd.DataFrame()
        self.df_niq_raw = pd.DataFrame()
        self.non_num_client = 0
        self.non_num_niq = 0
        self.drill_down_level = "None"
        self.input_data_type = "Empty"
        self.periodicity = "Empty"
        self.fact_selections = {}
    
    def set_data(self, df_client: pd.DataFrame, df_niq: pd.DataFrame, 
                 df_client_raw: pd.DataFrame, df_niq_raw: pd.DataFrame,
                 non_num_client: int, non_num_niq: int, drill_down_level: str = "None"):
        """Asignar datos validados al manager"""
        # Validar drill_down_level
        valid_levels = ["Total", "Channels", "Category", "Brand", "Product", "None"]
        if drill_down_level not in valid_levels:
            raise ValueError(f"drill_down_level debe ser uno de: {valid_levels}")
        
        self.df_client = df_client.copy()
        self.df_niq = df_niq.copy()
        self.df_client_raw = df_client_raw.copy()
        self.df_niq_raw = df_niq_raw.copy()
        self.non_num_client = non_num_client
        self.non_num_niq = non_num_niq
        self.drill_down_level = drill_down_level
    
    def get_status(self) -> dict:
        """Obtener el estado actual del manager"""
        return {
            "df_client_shape": self.df_client.shape,
            "df_niq_shape": self.df_niq.shape,
            "df_client_raw_shape": self.df_client_raw.shape,
            "df_niq_raw_shape": self.df_niq_raw.shape,
            "non_num_client": self.non_num_client,
            "non_num_niq": self.non_num_niq,
            "drill_down_level": self.drill_down_level,
            "input_data_type": self.input_data_type,
            "periodicity": self.periodicity,
            "fact_selections": self.fact_selections,
            "has_data": not self.df_client.empty and not self.df_niq.empty,
            "has_raw_data": not self.df_client_raw.empty and not self.df_niq_raw.empty
        }
    
    def set_processing_params(self, input_data_type: str, periodicity: str):
        """Asignar parámetros de procesamiento con validación"""
        valid_data_types = ["Value", "Grams", "Kilograms", "Liters", "Milliliters", "Empty"]
        valid_periodicities = ["monthly", "bimonthly", "Empty"]
        
        if input_data_type not in valid_data_types:
            raise ValueError(f"input_data_type debe ser uno de: {valid_data_types}")
        
        if periodicity not in valid_periodicities:
            raise ValueError(f"periodicity debe ser uno de: {valid_periodicities}")
        
        self.input_data_type = input_data_type
        self.periodicity = periodicity

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear instancia global del DataManager (Singleton)
data_manager = DataManager()

# Crear la instancia de FastAPI
app = FastAPI(
    title="Excel Validation API",
    description="API para validar archivos Excel de Cliente y NIQ",
    version="1.0.0"
)

# Configurar CORS para permitir requests desde React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://192.168.1.7:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "Excel Validation API",
        "version": "1.0.0",
        "status": "running",
        "available_endpoints": [
            "GET / - Información de la API",
            "POST /validate-excel-files - Validar archivos Cliente + NIQ",
            "GET /data-manager-status - Consultar estado del DataManager",
            "GET /export-client-excel - Exportar df_client (procesado) como Excel",
            "GET /export-niq-excel - Exportar df_niq (procesado) como Excel",
            "GET /export-client-raw-excel - Exportar df_client_raw (original) como Excel",
            "GET /export-niq-raw-excel - Exportar df_niq_raw (original) como Excel",
            "POST /calculate-coverage-total - Calcular cobertura a nivel total",
            "POST /calculate-coverage-channels - Calcular cobertura a nivel de canales (DTT y Modern)",
            "POST /calculate-coverage-brands - Calcular cobertura a nivel de marcas",
            "POST /unify-channels-brands - Unificar canales y marcas con homologación"
        ]
    }

@app.get("/data-manager-status")
async def get_data_manager_status():
    """
    Consultar el estado actual del DataManager
    
    Retorna información sobre los datos cargados actualmente
    """
    try:
        status = data_manager.get_status()
        return {
            "success": True,
            "data_manager_status": status,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error consultando estado del DataManager: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error consultando estado: {str(e)}")

@app.get("/export-client-excel")
async def export_client_excel():
    """
    Exporta data_manager.df_client como archivo Excel para descarga
    
    Returns:
        Archivo Excel con los datos del cliente
    """
    try:
        logger.info("=== Iniciando exportación de df_client a Excel ===")
        
        # Validar que hay datos en df_client
        if data_manager.df_client.empty:
            raise HTTPException(
                status_code=400, 
                detail="No hay datos de cliente cargados en el DataManager. Primero ejecute validate-excel-files."
            )
        
        # Crear un buffer en memoria para el archivo Excel
        output = io.BytesIO()
        
        # Escribir el DataFrame a Excel en el buffer
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data_manager.df_client.to_excel(writer, sheet_name='Client_Data', index=False)
        
        # Mover el puntero al inicio del buffer
        output.seek(0)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client_data_{timestamp}.xlsx"
        
        logger.info(f"Exportación de df_client completada. Filas: {len(data_manager.df_client)}, Columnas: {len(data_manager.df_client.columns)}")
        
        # Retornar el archivo como respuesta de streaming
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando df_client a Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando datos del cliente: {str(e)}")

@app.get("/export-niq-excel")
async def export_niq_excel():
    """
    Exporta data_manager.df_niq como archivo Excel para descarga
    
    Returns:
        Archivo Excel con los datos de NIQ
    """
    try:
        logger.info("=== Iniciando exportación de df_niq a Excel ===")
        
        # Validar que hay datos en df_niq
        if data_manager.df_niq.empty:
            raise HTTPException(
                status_code=400, 
                detail="No hay datos de NIQ cargados en el DataManager. Primero ejecute validate-excel-files."
            )
        
        # Crear un buffer en memoria para el archivo Excel
        output = io.BytesIO()
        
        # Escribir el DataFrame a Excel en el buffer
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data_manager.df_niq.to_excel(writer, sheet_name='NIQ_Data', index=False)
        
        # Mover el puntero al inicio del buffer
        output.seek(0)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"niq_data_{timestamp}.xlsx"
        
        logger.info(f"Exportación de df_niq completada. Filas: {len(data_manager.df_niq)}, Columnas: {len(data_manager.df_niq.columns)}")
        
        # Retornar el archivo como respuesta de streaming
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando df_niq a Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando datos de NIQ: {str(e)}")

@app.get("/export-client-raw-excel")
async def export_client_raw_excel():
    """
    Exporta data_manager.df_client_raw (datos originales sin NLP) como archivo Excel para descarga
    
    Returns:
        Archivo Excel con los datos originales del cliente (sin procesamiento NLP)
    """
    try:
        logger.info("=== Iniciando exportación de df_client_raw a Excel ===")
        
        # Validar que hay datos en df_client_raw
        if data_manager.df_client_raw.empty:
            raise HTTPException(
                status_code=400, 
                detail="No hay datos raw de cliente cargados en el DataManager. Primero ejecute validate-excel-files."
            )
        
        # Crear un buffer en memoria para el archivo Excel
        output = io.BytesIO()
        
        # Escribir el DataFrame a Excel en el buffer
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data_manager.df_client_raw.to_excel(writer, sheet_name='Client_Data_Raw', index=False)
        
        # Mover el puntero al inicio del buffer
        output.seek(0)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client_data_raw_{timestamp}.xlsx"
        
        logger.info(f"Exportación de df_client_raw completada. Filas: {len(data_manager.df_client_raw)}, Columnas: {len(data_manager.df_client_raw.columns)}")
        
        # Retornar el archivo como respuesta de streaming
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando df_client_raw a Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando datos raw del cliente: {str(e)}")

@app.get("/export-niq-raw-excel")
async def export_niq_raw_excel():
    """
    Exporta data_manager.df_niq_raw (datos originales) como archivo Excel para descarga
    
    Returns:
        Archivo Excel con los datos originales de NIQ
    """
    try:
        logger.info("=== Iniciando exportación de df_niq_raw a Excel ===")
        
        # Validar que hay datos en df_niq_raw
        if data_manager.df_niq_raw.empty:
            raise HTTPException(
                status_code=400, 
                detail="No hay datos raw de NIQ cargados en el DataManager. Primero ejecute validate-excel-files."
            )
        
        # Crear un buffer en memoria para el archivo Excel
        output = io.BytesIO()
        
        # Escribir el DataFrame a Excel en el buffer
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data_manager.df_niq_raw.to_excel(writer, sheet_name='NIQ_Data_Raw', index=False)
        
        # Mover el puntero al inicio del buffer
        output.seek(0)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"niq_data_raw_{timestamp}.xlsx"
        
        logger.info(f"Exportación de df_niq_raw completada. Filas: {len(data_manager.df_niq_raw)}, Columnas: {len(data_manager.df_niq_raw.columns)}")
        
        # Retornar el archivo como respuesta de streaming
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando df_niq_raw a Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando datos raw de NIQ: {str(e)}")

@app.post("/calculate-coverage-total")
async def calculate_coverage_total(request: CoverageTotalRequest):
    """
    Calcula la cobertura a nivel total y retorna información para scorecard y gráfica de tendencias.
    
    Args:
        request: Objeto con drill_down_level (Total, Channels, Category, Brand, Product, None)
    
    Returns:
        Dict con scorecard y datos de tendencia para gráficas
    """
    try:
        logger.info(f"=== Iniciando cálculo de cobertura total con drill_down_level: {request.drill_down_level} ===")
        
        # Validar que hay datos cargados
        if data_manager.df_client_raw.empty or data_manager.df_niq_raw.empty or data_manager.df_client.empty:
            return {"error": "No hay datos cargados en el DataManager. Primero ejecute validate-excel-files."}
        
        # Validar que hay fact_mapping
        if not data_manager.fact_selections:
            return {"error": "No hay fact_mapping configurado. Primero ejecute unify-channels-brands."}
        
        # 0. Hacer copias locales
        df_client_copy = data_manager.df_client.copy()
        df_niq_copy = data_manager.df_niq.copy()
        df_niq_raw_copy = data_manager.df_niq_raw.copy()
        
        logger.info(f"Copias creadas - Cliente: {df_client_copy.shape}, NIQ raw: {df_niq_copy.shape}")
        
        # 1 y 2. Igualar columnas numéricas y limpiar filas nulas (usando función de utilidad)
        try:
            niq_num = df_niq_copy.select_dtypes(include=[np.number]).shape[1]
            client_num = df_client_copy.select_dtypes(include=[np.number]).shape[1]
            
            # Aplicar align_numeric_and_clean
            df_client_copy, df_niq_copy = align_numeric_and_clean(
                df_client=df_client_copy,
                df_niq=df_niq_copy,
                periodicity=data_manager.periodicity,
                niq_num_count=niq_num,
                client_num_count=client_num,
            )
            
        except Exception as e:
            logger.error(f"Error en align_numeric_and_clean: {str(e)}")
            return {"error": f"Error en pasos 1-2 (igualar y limpiar): {str(e)}"}
        
        # 3. Filtrar por fact y categoría (usando función de utilidad)
        try:
            df_client_copy, df_niq_copy = filter_by_fact_and_group(
                df_client=df_client_copy,
                df_niq=df_niq_copy,
                fact_sales_name=data_manager.fact_selections.get("Sales", ""),
                column_index_filter_niq=0,  # Manufacturer (posición 0)
                column_index_filter_client=2,  # Country (posición 2)
                input_data_type=data_manager.input_data_type,
            )
            
        except ValueError as e:
            logger.error(f"Error en filter_by_fact_and_group: {str(e)}")
            return {"error": f"Error en paso 3 (filtrar y agrupar): {str(e)}"}
        
        # 4. Aplicar periodicidad y extraer WD, ND, Share
        try:
            # Extraer métricas de NIQ
            nd, wd, share = extract_metrics_from_niq(
                df_niq_raw_copy=df_niq_raw_copy,
                drill_down_level=request.drill_down_level,
                nd_fact_name=data_manager.fact_selections.get("Numeric Distribution (ND)", ""),
                wd_fact_name=data_manager.fact_selections.get("Weighted Distribution (WD)", ""),
                share_fact_name=data_manager.fact_selections.get("Share", ""),
                non_num_niq=data_manager.non_num_niq
            )
            
            # Agrupar columnas si es bimensual
            if data_manager.periodicity == 'bimonthly':
                # Para df_niq_copy
                numeric_cols = df_niq_copy.select_dtypes(include=[np.number]).columns.tolist()
                new_cols = {}
                for i in range(0, len(numeric_cols), 2):
                    if i + 1 < len(numeric_cols):
                        new_col_name = f"{numeric_cols[i]}/{numeric_cols[i+1]}"
                        new_cols[new_col_name] = df_niq_copy[numeric_cols[i]] + df_niq_copy[numeric_cols[i+1]]
                
                df_niq_copy = df_niq_copy.drop(columns=numeric_cols)
                for col, values in new_cols.items():
                    df_niq_copy[col] = values
                
                # Para df_client_copy
                numeric_cols = df_client_copy.select_dtypes(include=[np.number]).columns.tolist()
                new_cols = {}
                for i in range(0, len(numeric_cols), 2):
                    if i + 1 < len(numeric_cols):
                        new_col_name = f"{numeric_cols[i]}/{numeric_cols[i+1]}"
                        new_cols[new_col_name] = df_client_copy[numeric_cols[i]] + df_client_copy[numeric_cols[i+1]]
                
                df_client_copy = df_client_copy.drop(columns=numeric_cols)
                for col, values in new_cols.items():
                    df_client_copy[col] = values
                
                logger.info("Columnas agrupadas para periodicidad bimensual")
            
        except Exception as e:
            logger.error(f"Error aplicando periodicidad: {str(e)}")
            return {"error": f"Error en paso 4 (periodicidad): {str(e)}"}
        
        # 5. Calcular cobertura
        try:
            niq_numeric_cols = df_niq_copy.select_dtypes(include=[np.number]).columns.tolist()
            client_numeric_cols = df_client_copy.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(niq_numeric_cols) == 0 or len(client_numeric_cols) == 0:
                return {"error": "No hay columnas numéricas para calcular cobertura"}
            
            # Determinar períodos para cálculo de cobertura (1 año)
            periods_per_year = 12 if data_manager.periodicity == 'monthly' else 6
            
            coverages = []
            niq_values = []
            client_values = []
            period_names = []
            
            # Obtener valores de NIQ y Cliente
            niq_row_values = df_niq_copy.iloc[0][niq_numeric_cols].values if len(df_niq_copy) > 0 else []
            client_row_values = df_client_copy.iloc[0][client_numeric_cols].values if len(df_client_copy) > 0 else []
            
            # Calcular cobertura para cada período válido
            for i in range(len(niq_numeric_cols)):
                period_names.append(niq_numeric_cols[i])
                niq_values.append(float(niq_row_values[i]) if i < len(niq_row_values) else 0)
                client_values.append(float(client_row_values[i]) if i < len(client_row_values) else 0)
                
                # Solo calcular cobertura si hay suficientes datos (1 año)
                if i >= periods_per_year:
                    start_idx = i - periods_per_year
                    end_idx = i
                    
                    niq_sum = sum(niq_row_values[start_idx:end_idx+1])
                    client_sum = sum(client_row_values[start_idx:end_idx+1])
                    
                    if client_sum > 0:
                        coverage = round((niq_sum / client_sum) * 100, 2)
                    else:
                        coverage = 0
                    
                    coverages.append(coverage)
                else:
                    coverages.append(0)  # No hay suficientes datos para 1 año
            
            logger.info(f"Cobertura calculada para {len(coverages)} períodos")
            
        except Exception as e:
            logger.error(f"Error calculando cobertura: {str(e)}")
            return {"error": f"Error en paso 5 (calcular cobertura): {str(e)}"}
        
        # 6. Construir respuesta
        try:
            # 6.1 Scorecard
            manufacturer = "Unknown"
            manufacts = df_niq_raw_copy.iloc[:, 0].dropna().unique().tolist()
            if len(manufacts) > 0 and len(df_niq_raw_copy.columns) > 0:
                manufacturer = manufacts[0]
            time_frame = period_names[-1] if period_names else "N/A"
            
            # latest_mat y mat_yago
            latest_mat = coverages[-1] if coverages else 0
            mat_yago = coverages[-1 - periods_per_year] if len(coverages) > periods_per_year else 0
            
            difference = round(latest_mat - mat_yago, 2)
            
            # Tendencias
            trend_niq = 99.9  # Fallback
            trend_pepsico = 99.9  # Fallback
            trend_error_message = None
            
            try:
                # Calcular trend_niq: (suma último año NIQ / suma año anterior NIQ) - 1
                # Verificar que hay suficientes datos (al menos 2 años)
                if len(niq_values) >= periods_per_year * 2:
                    # Último año: últimos periods_per_year valores
                    niq_last_year_sum = sum(niq_values[-periods_per_year:])
                    # Año anterior: los periods_per_year valores antes del último año
                    niq_previous_year_sum = sum(niq_values[-periods_per_year * 2:-periods_per_year])
                    
                    if niq_previous_year_sum > 0:
                        trend_niq = round(((niq_last_year_sum / niq_previous_year_sum) - 1) * 100, 2)
                        logger.info(f"Trend NIQ calculado: {trend_niq}% (último año: {niq_last_year_sum}, año anterior: {niq_previous_year_sum})")
                    else:
                        trend_error_message = "No se pudo calcular trend_niq: suma del año anterior es 0"
                        logger.warning(trend_error_message)
                else:
                    trend_error_message = f"No hay suficientes datos para calcular trend_niq: se necesitan {periods_per_year * 2} períodos, hay {len(niq_values)}"
                    logger.warning(trend_error_message)
                
                # Calcular trend_pepsico: (suma último año Cliente / suma año anterior Cliente) - 1
                if len(client_values) >= periods_per_year * 2:
                    # Último año: últimos periods_per_year valores
                    client_last_year_sum = sum(client_values[-periods_per_year:])
                    # Año anterior: los periods_per_year valores antes del último año
                    client_previous_year_sum = sum(client_values[-periods_per_year * 2:-periods_per_year])
                    
                    if client_previous_year_sum > 0:
                        trend_pepsico = round(((client_last_year_sum / client_previous_year_sum) - 1) * 100, 2)
                        logger.info(f"Trend Cliente calculado: {trend_pepsico}% (último año: {client_last_year_sum}, año anterior: {client_previous_year_sum})")
                    else:
                        trend_error_message = "No se pudo calcular trend_pepsico: suma del año anterior es 0"
                        logger.warning(trend_error_message)
                else:
                    if not trend_error_message:  # Solo si no hubo error anterior
                        trend_error_message = f"No hay suficientes datos para calcular trend_pepsico: se necesitan {periods_per_year * 2} períodos, hay {len(client_values)}"
                    logger.warning(trend_error_message)
                    
            except Exception as e:
                trend_error_message = f"Error calculando tendencias: {str(e)}"
                logger.error(trend_error_message)
                # Mantener valores fallback (99.9)
            
            scorecard = {
                "manufacturer": manufacturer,
                "time_frame": time_frame,
                "mat_yago": mat_yago,
                "latest_mat": latest_mat,
                "difference": difference,
                "trend_niq": trend_niq,
                "trend_pepsico": trend_pepsico,
                "wd": wd,
                "nd": nd,
                "share": share,
                "drill_down_level": request.drill_down_level
            }
            
            # Agregar mensaje de error si hubo problemas calculando tendencias
            if trend_error_message:
                scorecard["trend_warning"] = trend_error_message
            
            # 6.2 Datos de gráfica
            chart_data = []
            for i in range(len(period_names)):
                chart_data.append({
                    "period": period_names[i],
                    "coverage": coverages[i],
                    "nielseniq": round(niq_values[i], 2),
                    "client": round(client_values[i], 2)
                })
            
            # Metadata
            date_range = f"{period_names[0]} - {period_names[-1]}" if period_names else "N/A"
            
            response = {
                "scorecard": scorecard,
                "chart": {
                    "title": "Market Coverage Trends",
                    "description": "Market trends for the selected period",
                    "data": chart_data,
                    "metadata": {
                        "max_months": len(period_names),
                        "date_range": date_range
                    }
                }
            }
            
            logger.info("Cálculo de cobertura completado exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"Error construyendo respuesta: {str(e)}")
            return {"error": f"Error en paso 6 (construir respuesta): {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error general en calculate_coverage_total: {str(e)}")
        return {"error": f"Error general: {str(e)}"}


@app.post("/calculate-coverage-channels")
async def calculate_coverage_channels(request: CoverageChannelsRequest):
    """
    Calcula la cobertura a nivel de canales (DTT y Modern) y retorna información para scorecard y gráfica de tendencias.
    
    Args:
        request: Objeto con drill_down_level
    
    Returns:
        Dict con scorecards y datos de tendencia para gráficas por canal (DTT y Modern)
    """
    try:
        logger.info(f"=== Iniciando cálculo de cobertura por canales con drill_down_level: {request.drill_down_level} ===")
        
        # Validar que hay datos cargados
        if data_manager.df_client_raw.empty or data_manager.df_niq_raw.empty or data_manager.df_client.empty or data_manager.df_niq.empty:
            return {"error": "No hay datos cargados en el DataManager. Primero ejecute validate-excel-files."}
        
        # Validar que hay fact_mapping
        if not data_manager.fact_selections:
            return {"error": "No hay fact_mapping configurado. Primero ejecute unify-channels-brands."}
        
        # Validar que existe la columna Homologado_C
        if 'Homologado_C' not in data_manager.df_client.columns or 'Homologado_C' not in data_manager.df_niq.columns:
            return {"error": "No existe la columna Homologado_C. Primero ejecute unify-channels-brands."}
        
        # 0. Hacer copias locales
        df_client_copy = data_manager.df_client.copy()
        df_niq_copy = data_manager.df_niq.copy()
        df_niq_raw_copy = data_manager.df_niq_raw.copy()
        
        logger.info(f"Copias creadas - Cliente: {df_client_copy.shape}, NIQ: {df_niq_copy.shape}")
        
        # 1 y 2. Igualar columnas numéricas y limpiar filas nulas (usando función de utilidad)
        try:
            niq_num = df_niq_copy.select_dtypes(include=[np.number]).shape[1]
            client_num = df_client_copy.select_dtypes(include=[np.number]).shape[1]
            
            # Aplicar align_numeric_and_clean (limpiar sobre NIQ procesado, no raw)
            df_client_copy, df_niq_copy = align_numeric_and_clean(
                df_client=df_client_copy,
                df_niq=df_niq_copy,
                periodicity=data_manager.periodicity,
                niq_num_count=niq_num,
                client_num_count=client_num,
            )
            
        except Exception as e:
            logger.error(f"Error en align_numeric_and_clean: {str(e)}")
            return {"error": f"Error en pasos 1-2 (igualar y limpiar): {str(e)}"}
        
        # 3. Filtrar por fact y canal (usando función de utilidad)
        try:
            # Obtener el índice de la columna 'Homologado_C' para ambos DataFrames
            idx_channel_niq = df_niq_copy.columns.get_loc("Homologado_C") if "Homologado_C" in df_niq_copy.columns else -1
            idx_channel_client = df_client_copy.columns.get_loc("Homologado_C") if "Homologado_C" in df_client_copy.columns else -1
            
            if idx_channel_niq == -1 or idx_channel_client == -1:
                return {"error": "No se encontró la columna 'Homologado_C' en los DataFrames. Ejecute unify-channels-brands primero."}
            
            df_client_copy, df_niq_copy = filter_by_fact_and_group(
                df_client=df_client_copy,
                df_niq=df_niq_copy,
                fact_sales_name=data_manager.fact_selections.get("Sales", ""),
                column_index_filter_niq=idx_channel_niq,  # Homologado_C
                column_index_filter_client=idx_channel_client,  # Homologado_C
                input_data_type=data_manager.input_data_type,
            )
            
        except ValueError as e:
            logger.error(f"Error en filter_by_fact_and_group: {str(e)}")
            return {"error": f"Error en paso 3 (filtrar y agrupar): {str(e)}"}
        except Exception as e:
            logger.error(f"Error inesperado en filter_by_fact_and_group: {str(e)}")
            return {"error": f"Error en paso 3 (filtrar y agrupar): {str(e)}"}
        
        # 4. Aplicar periodicidad
        try:
            # Extraer métricas de NIQ
            nd, wd, share = extract_metrics_from_niq(
                df_niq_raw_copy=df_niq_raw_copy,
                drill_down_level=request.drill_down_level,
                nd_fact_name=data_manager.fact_selections.get("Numeric Distribution (ND)", ""),
                wd_fact_name=data_manager.fact_selections.get("Weighted Distribution (WD)", ""),
                share_fact_name=data_manager.fact_selections.get("Share", ""),
                non_num_niq=data_manager.non_num_niq
            )
            
            # Agrupar columnas si es bimensual
            if data_manager.periodicity == 'bimonthly':
                # Para df_niq_copy
                numeric_cols = df_niq_copy.select_dtypes(include=[np.number]).columns.tolist()
                new_cols = {}
                for i in range(0, len(numeric_cols), 2):
                    if i + 1 < len(numeric_cols):
                        new_col_name = f"{numeric_cols[i]}/{numeric_cols[i+1]}"
                        new_cols[new_col_name] = df_niq_copy[numeric_cols[i]] + df_niq_copy[numeric_cols[i+1]]
                
                df_niq_copy = df_niq_copy.drop(columns=numeric_cols)
                for col, values in new_cols.items():
                    df_niq_copy[col] = values
                
                # Para df_client_copy
                numeric_cols = df_client_copy.select_dtypes(include=[np.number]).columns.tolist()
                new_cols = {}
                for i in range(0, len(numeric_cols), 2):
                    if i + 1 < len(numeric_cols):
                        new_col_name = f"{numeric_cols[i]}/{numeric_cols[i+1]}"
                        new_cols[new_col_name] = df_client_copy[numeric_cols[i]] + df_client_copy[numeric_cols[i+1]]
                
                df_client_copy = df_client_copy.drop(columns=numeric_cols)
                for col, values in new_cols.items():
                    df_client_copy[col] = values
                
                logger.info("Columnas agrupadas para periodicidad bimensual")
            
        except Exception as e:
            logger.error(f"Error aplicando periodicidad: {str(e)}")
            return {"error": f"Error en paso 4 (periodicidad): {str(e)}"}
        
        # 5. Calcular cobertura y tendencias por canal
        try:
            # Separar por canales
            df_client_dtt = df_client_copy[df_client_copy['Homologado_C'] == 'DTT (Traditional)'].copy()
            df_client_modern = df_client_copy[df_client_copy['Homologado_C'] == 'Modern Trade'].copy()
            df_niq_dtt = df_niq_copy[df_niq_copy['Homologado_C'] == 'DTT (Traditional)'].copy()
            df_niq_modern = df_niq_copy[df_niq_copy['Homologado_C'] == 'Modern Trade'].copy()
            
            logger.info(f"Datos separados - DTT: NIQ={len(df_niq_dtt)}, Cliente={len(df_client_dtt)} | Modern: NIQ={len(df_niq_modern)}, Cliente={len(df_client_modern)}")
            
            # Agrupar por canal (sumar todas las filas dentro de cada canal)
            if len(df_niq_dtt) > 0:
                df_niq_dtt = df_niq_dtt.groupby('Homologado_C', as_index=False).sum(numeric_only=True)
                logger.info(f"df_niq_dtt después de agrupar: {df_niq_dtt.shape}")
            
            if len(df_niq_modern) > 0:
                df_niq_modern = df_niq_modern.groupby('Homologado_C', as_index=False).sum(numeric_only=True)
                logger.info(f"df_niq_modern después de agrupar: {df_niq_modern.shape}")
            
            if len(df_client_dtt) > 0:
                df_client_dtt = df_client_dtt.groupby('Homologado_C', as_index=False).sum(numeric_only=True)
                logger.info(f"df_client_dtt después de agrupar: {df_client_dtt.shape}")
            
            if len(df_client_modern) > 0:
                df_client_modern = df_client_modern.groupby('Homologado_C', as_index=False).sum(numeric_only=True)
                logger.info(f"df_client_modern después de agrupar: {df_client_modern.shape}")
            
            periods_per_year = 12 if data_manager.periodicity == 'monthly' else 6
            
            # ===== CALCULAR MÉTRICAS PARA DTT =====
            dtt_coverages = []
            dtt_niq_values = []
            dtt_client_values = []
            dtt_period_names = []
            dtt_trend_niq = 99.9
            dtt_trend_client = 99.9
            dtt_error = None
            
            try:
                if len(df_niq_dtt) == 0 or len(df_client_dtt) == 0:
                    dtt_error = "No hay datos para el canal DTT"
                    logger.warning(dtt_error)
                else:
                    dtt_niq_numeric_cols = df_niq_dtt.select_dtypes(include=[np.number]).columns.tolist()
                    dtt_client_numeric_cols = df_client_dtt.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(dtt_niq_numeric_cols) == 0 or len(dtt_client_numeric_cols) == 0:
                        dtt_error = "No hay columnas numéricas para el canal DTT"
                        logger.warning(dtt_error)
                    else:
                        # Obtener valores de NIQ y Cliente para DTT
                        dtt_niq_row_values = df_niq_dtt.iloc[0][dtt_niq_numeric_cols].values
                        dtt_client_row_values = df_client_dtt.iloc[0][dtt_client_numeric_cols].values
                        
                        # Calcular cobertura para cada período en DTT
                        for i in range(len(dtt_niq_numeric_cols)):
                            dtt_period_names.append(dtt_niq_numeric_cols[i])
                            dtt_niq_values.append(float(dtt_niq_row_values[i]) if i < len(dtt_niq_row_values) else 0)
                            dtt_client_values.append(float(dtt_client_row_values[i]) if i < len(dtt_client_row_values) else 0)
                            
                            # Solo calcular cobertura si hay suficientes datos (1 año)
                            if i >= periods_per_year:
                                start_idx = i - periods_per_year
                                end_idx = i
                                
                                dtt_niq_sum = sum(dtt_niq_row_values[start_idx:end_idx+1])
                                dtt_client_sum = sum(dtt_client_row_values[start_idx:end_idx+1])
                                
                                if dtt_client_sum > 0:
                                    dtt_coverage = round((dtt_niq_sum / dtt_client_sum) * 100, 2)
                                else:
                                    dtt_coverage = 0
                                
                                dtt_coverages.append(dtt_coverage)
                            else:
                                dtt_coverages.append(0)
                        
                        # Calcular tendencias para DTT
                        if len(dtt_niq_values) >= periods_per_year * 2:
                            dtt_niq_last_year_sum = sum(dtt_niq_values[-periods_per_year:])
                            dtt_niq_previous_year_sum = sum(dtt_niq_values[-periods_per_year * 2:-periods_per_year])
                            
                            if dtt_niq_previous_year_sum > 0:
                                dtt_trend_niq = round(((dtt_niq_last_year_sum / dtt_niq_previous_year_sum) - 1) * 100, 2)
                                logger.info(f"Trend NIQ DTT calculado: {dtt_trend_niq}%")
                        
                        if len(dtt_client_values) >= periods_per_year * 2:
                            dtt_client_last_year_sum = sum(dtt_client_values[-periods_per_year:])
                            dtt_client_previous_year_sum = sum(dtt_client_values[-periods_per_year * 2:-periods_per_year])
                            
                            if dtt_client_previous_year_sum > 0:
                                dtt_trend_client = round(((dtt_client_last_year_sum / dtt_client_previous_year_sum) - 1) * 100, 2)
                                logger.info(f"Trend Cliente DTT calculado: {dtt_trend_client}%")
                        
                        logger.info(f"Métricas calculadas para DTT: {len(dtt_coverages)} períodos")
                        
            except Exception as e:
                dtt_error = f"Error calculando métricas para DTT: {str(e)}"
                logger.error(dtt_error)
            
            # ===== CALCULAR MÉTRICAS PARA MODERN =====
            modern_coverages = []
            modern_niq_values = []
            modern_client_values = []
            modern_period_names = []
            modern_trend_niq = 99.9
            modern_trend_client = 99.9
            modern_error = None
            
            try:
                if len(df_niq_modern) == 0 or len(df_client_modern) == 0:
                    modern_error = "No hay datos para el canal Modern"
                    logger.warning(modern_error)
                else:
                    modern_niq_numeric_cols = df_niq_modern.select_dtypes(include=[np.number]).columns.tolist()
                    modern_client_numeric_cols = df_client_modern.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(modern_niq_numeric_cols) == 0 or len(modern_client_numeric_cols) == 0:
                        modern_error = "No hay columnas numéricas para el canal Modern"
                        logger.warning(modern_error)
                    else:
                        # Obtener valores de NIQ y Cliente para Modern
                        modern_niq_row_values = df_niq_modern.iloc[0][modern_niq_numeric_cols].values
                        modern_client_row_values = df_client_modern.iloc[0][modern_client_numeric_cols].values
                        
                        # Calcular cobertura para cada período en Modern
                        for i in range(len(modern_niq_numeric_cols)):
                            modern_period_names.append(modern_niq_numeric_cols[i])
                            modern_niq_values.append(float(modern_niq_row_values[i]) if i < len(modern_niq_row_values) else 0)
                            modern_client_values.append(float(modern_client_row_values[i]) if i < len(modern_client_row_values) else 0)
                            
                            # Solo calcular cobertura si hay suficientes datos (1 año)
                            if i >= periods_per_year:
                                start_idx = i - periods_per_year
                                end_idx = i
                                
                                modern_niq_sum = sum(modern_niq_row_values[start_idx:end_idx+1])
                                modern_client_sum = sum(modern_client_row_values[start_idx:end_idx+1])
                                
                                if modern_client_sum > 0:
                                    modern_coverage = round((modern_niq_sum / modern_client_sum) * 100, 2)
                                else:
                                    modern_coverage = 0
                                
                                modern_coverages.append(modern_coverage)
                            else:
                                modern_coverages.append(0)
                        
                        # Calcular tendencias para Modern
                        if len(modern_niq_values) >= periods_per_year * 2:
                            modern_niq_last_year_sum = sum(modern_niq_values[-periods_per_year:])
                            modern_niq_previous_year_sum = sum(modern_niq_values[-periods_per_year * 2:-periods_per_year])
                            
                            if modern_niq_previous_year_sum > 0:
                                modern_trend_niq = round(((modern_niq_last_year_sum / modern_niq_previous_year_sum) - 1) * 100, 2)
                                logger.info(f"Trend NIQ Modern calculado: {modern_trend_niq}%")
                        
                        if len(modern_client_values) >= periods_per_year * 2:
                            modern_client_last_year_sum = sum(modern_client_values[-periods_per_year:])
                            modern_client_previous_year_sum = sum(modern_client_values[-periods_per_year * 2:-periods_per_year])
                            
                            if modern_client_previous_year_sum > 0:
                                modern_trend_client = round(((modern_client_last_year_sum / modern_client_previous_year_sum) - 1) * 100, 2)
                                logger.info(f"Trend Cliente Modern calculado: {modern_trend_client}%")
                        
                        logger.info(f"Métricas calculadas para Modern: {len(modern_coverages)} períodos")
                        
            except Exception as e:
                modern_error = f"Error calculando métricas para Modern: {str(e)}"
                logger.error(modern_error)
            
        except Exception as e:
            logger.error(f"Error calculando cobertura por canal: {str(e)}")
            return {"error": f"Error en paso 5 (calcular cobertura): {str(e)}"}
        
        # 6. Construir respuesta
        try:
            # Obtener manufacturer
            manufacturer = "Unknown"
            manufacts = df_niq_raw_copy.iloc[:, 0].dropna().unique().tolist()
            if len(manufacts) > 0 and len(df_niq_raw_copy.columns) > 0:
                manufacturer = manufacts[0]
            
            # ===== CONSTRUIR RESPUESTA PARA DTT =====
            dtt_response = []
            
            if dtt_error:
                dtt_response = [{"error": dtt_error}, {"error": dtt_error}]
            else:
                # Scorecard DTT
                dtt_time_frame = dtt_period_names[-1] if dtt_period_names else "N/A"
                dtt_latest_mat = dtt_coverages[-1] if dtt_coverages else 0
                dtt_mat_yago = dtt_coverages[-1 - periods_per_year] if len(dtt_coverages) > periods_per_year else 0
                dtt_difference = round(dtt_latest_mat - dtt_mat_yago, 2)
                
                dtt_scorecard = {
                    "manufacturer": manufacturer,
                    "channel": "DTT",
                    "time_frame": dtt_time_frame,
                    "mat_yago": dtt_mat_yago,
                    "latest_mat": dtt_latest_mat,
                    "difference": dtt_difference,
                    "trend_niq": dtt_trend_niq,
                    "trend_pepsico": dtt_trend_client,
                    "wd": wd,
                    "nd": nd,
                    "share": share,
                    "drill_down_level": request.drill_down_level
                }
                
                # Chart DTT
                dtt_chart_data = []
                for i in range(len(dtt_period_names)):
                    dtt_chart_data.append({
                        "period": dtt_period_names[i],
                        "coverage": dtt_coverages[i],
                        "nielseniq": round(dtt_niq_values[i], 2),
                        "client": round(dtt_client_values[i], 2)
                    })
                
                dtt_date_range = f"{dtt_period_names[0]} - {dtt_period_names[-1]}" if dtt_period_names else "N/A"
                
                dtt_chart = {
                    "title": "Market Coverage Trends - DTT",
                    "description": "Market trends for DTT channel",
                    "data": dtt_chart_data,
                    "metadata": {
                        "max_months": len(dtt_period_names),
                        "date_range": dtt_date_range
                    }
                }
                
                dtt_response = [dtt_scorecard, dtt_chart]
            
            # ===== CONSTRUIR RESPUESTA PARA MODERN =====
            modern_response = []
            
            if modern_error:
                modern_response = [{"error": modern_error}, {"error": modern_error}]
            else:
                # Scorecard Modern
                modern_time_frame = modern_period_names[-1] if modern_period_names else "N/A"
                modern_latest_mat = modern_coverages[-1] if modern_coverages else 0
                modern_mat_yago = modern_coverages[-1 - periods_per_year] if len(modern_coverages) > periods_per_year else 0
                modern_difference = round(modern_latest_mat - modern_mat_yago, 2)
                
                modern_scorecard = {
                    "manufacturer": manufacturer,
                    "channel": "Modern",
                    "time_frame": modern_time_frame,
                    "mat_yago": modern_mat_yago,
                    "latest_mat": modern_latest_mat,
                    "difference": modern_difference,
                    "trend_niq": modern_trend_niq,
                    "trend_pepsico": modern_trend_client,
                    "wd": wd,
                    "nd": nd,
                    "share": share,
                    "drill_down_level": request.drill_down_level
                }
                
                # Chart Modern
                modern_chart_data = []
                for i in range(len(modern_period_names)):
                    modern_chart_data.append({
                        "period": modern_period_names[i],
                        "coverage": modern_coverages[i],
                        "nielseniq": round(modern_niq_values[i], 2),
                        "client": round(modern_client_values[i], 2)
                    })
                
                modern_date_range = f"{modern_period_names[0]} - {modern_period_names[-1]}" if modern_period_names else "N/A"
                
                modern_chart = {
                    "title": "Market Coverage Trends - Modern",
                    "description": "Market trends for Modern channel",
                    "data": modern_chart_data,
                    "metadata": {
                        "max_months": len(modern_period_names),
                        "date_range": modern_date_range
                    }
                }
                
                modern_response = [modern_scorecard, modern_chart]
            
            # Respuesta final
            response = {
                "DTT": dtt_response,
                "Modern": modern_response
            }
            
            logger.info("Cálculo de cobertura por canales completado exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"Error construyendo respuesta: {str(e)}")
            return {"error": f"Error en paso 6 (construir respuesta): {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error general en calculate_coverage_channels: {str(e)}")
        return {"error": f"Error general: {str(e)}"}


@app.post("/calculate-coverage-brands")
async def calculate_coverage_brands(request: CoverageBrandsRequest):
    """
    Calcula la cobertura a nivel de marcas y retorna información para scorecard y gráfica de tendencias.
    
    Args:
        request: Objeto con drill_down_level y brand_names (lista de marcas a analizar)
    
    Returns:
        Dict con scorecards y datos de tendencia para gráficas por marca
    """
    try:
        logger.info(f"=== Iniciando cálculo de cobertura por marcas con drill_down_level: {request.drill_down_level} ===")
        logger.info(f"Marcas a analizar: {request.brand_names}")
        
        # Validar que hay datos cargados
        if data_manager.df_client_raw.empty or data_manager.df_niq_raw.empty or data_manager.df_client.empty or data_manager.df_niq.empty:
            return {"error": "No hay datos cargados en el DataManager. Primero ejecute validate-excel-files."}
        
        # Validar que hay fact_mapping
        if not data_manager.fact_selections:
            return {"error": "No hay fact_mapping configurado. Primero ejecute unify-channels-brands."}
        
        # Validar que existe la columna Homologado_B
        if 'Homologado_B' not in data_manager.df_client.columns:
            return {"error": "No existe la columna Homologado_B en df_client. Primero ejecute unify-channels-brands."}
        
        # Validar que se proporcionaron marcas
        if not request.brand_names or len(request.brand_names) == 0:
            return {"error": "Debe proporcionar al menos una marca en brand_names."}
        
        # 0. Hacer copias locales
        df_client_copy = data_manager.df_client.copy()
        df_niq_copy = data_manager.df_niq.copy()
        df_niq_raw_copy = data_manager.df_niq_raw.copy()
        
        logger.info(f"Copias creadas - Cliente: {df_client_copy.shape}, NIQ: {df_niq_copy.shape}")
        
        # 1 y 2. Igualar columnas numéricas y limpiar filas nulas (usando función de utilidad)
        try:
            niq_num = df_niq_copy.select_dtypes(include=[np.number]).shape[1]
            client_num = df_client_copy.select_dtypes(include=[np.number]).shape[1]
            
            # Aplicar align_numeric_and_clean
            df_client_copy, df_niq_copy = align_numeric_and_clean(
                df_client=df_client_copy,
                df_niq=df_niq_copy,
                periodicity=data_manager.periodicity,
                niq_num_count=niq_num,
                client_num_count=client_num,
            )
            
        except Exception as e:
            logger.error(f"Error en align_numeric_and_clean: {str(e)}")
            return {"error": f"Error en pasos 1-2 (igualar y limpiar): {str(e)}"}
        
        # 3. Filtrar por fact y marca (usando función de utilidad)
        try:
            # Obtener el índice de la columna 'Homologado_B'
            idx_brand_client = df_client_copy.columns.get_loc("Homologado_B") if "Homologado_B" in df_client_copy.columns else -1
            
            if idx_brand_client == -1:
                return {"error": "No se encontró la columna 'Homologado_B' en df_client. Ejecute unify-channels-brands primero."}
            
            df_client_copy, df_niq_copy = filter_by_fact_and_group(
                df_client=df_client_copy,
                df_niq=df_niq_copy,
                fact_sales_name=data_manager.fact_selections.get("Sales", ""),
                column_index_filter_niq=3,  # Brand (posición 3 en NIQ)
                column_index_filter_client=idx_brand_client,  # Homologado_B
                input_data_type=data_manager.input_data_type,
            )
            
        except ValueError as e:
            logger.error(f"Error en filter_by_fact_and_group: {str(e)}")
            return {"error": f"Error en paso 3 (filtrar y agrupar): {str(e)}"}
        except Exception as e:
            logger.error(f"Error inesperado en filter_by_fact_and_group: {str(e)}")
            return {"error": f"Error en paso 3 (filtrar y agrupar): {str(e)}"}
        
        # 4. Aplicar periodicidad
        try:
            # Extraer métricas de NIQ
            nd, wd, share = extract_metrics_from_niq(
                df_niq_raw_copy=df_niq_raw_copy,
                drill_down_level=request.drill_down_level,
                nd_fact_name=data_manager.fact_selections.get("Numeric Distribution (ND)", ""),
                wd_fact_name=data_manager.fact_selections.get("Weighted Distribution (WD)", ""),
                share_fact_name=data_manager.fact_selections.get("Share", ""),
                non_num_niq=data_manager.non_num_niq
            )
            
            # Agrupar columnas si es bimensual
            if data_manager.periodicity == 'bimonthly':
                # Para df_niq_copy
                numeric_cols = df_niq_copy.select_dtypes(include=[np.number]).columns.tolist()
                new_cols = {}
                for i in range(0, len(numeric_cols), 2):
                    if i + 1 < len(numeric_cols):
                        new_col_name = f"{numeric_cols[i]}/{numeric_cols[i+1]}"
                        new_cols[new_col_name] = df_niq_copy[numeric_cols[i]] + df_niq_copy[numeric_cols[i+1]]
                
                df_niq_copy = df_niq_copy.drop(columns=numeric_cols)
                for col, values in new_cols.items():
                    df_niq_copy[col] = values
                
                # Para df_client_copy
                numeric_cols = df_client_copy.select_dtypes(include=[np.number]).columns.tolist()
                new_cols = {}
                for i in range(0, len(numeric_cols), 2):
                    if i + 1 < len(numeric_cols):
                        new_col_name = f"{numeric_cols[i]}/{numeric_cols[i+1]}"
                        new_cols[new_col_name] = df_client_copy[numeric_cols[i]] + df_client_copy[numeric_cols[i+1]]
                
                df_client_copy = df_client_copy.drop(columns=numeric_cols)
                for col, values in new_cols.items():
                    df_client_copy[col] = values
                
                logger.info("Columnas agrupadas para periodicidad bimensual")
            
        except Exception as e:
            logger.error(f"Error aplicando periodicidad: {str(e)}")
            return {"error": f"Error en paso 4 (periodicidad): {str(e)}"}
        
        # 5. Calcular cobertura y tendencias por marca
        try:
            periods_per_year = 12 if data_manager.periodicity == 'monthly' else 6
            
            # Obtener manufacturer
            manufacturer = "Unknown"
            manufacts = df_niq_raw_copy.iloc[:, 0].dropna().unique().tolist()

            if len(manufacts) > 0 and len(df_niq_raw_copy.columns) > 0:
                manufacturer = manufacts[0]
            
            # Diccionario de resultados por marca
            results = {}
            
            # Procesar cada marca
            for brand_name in request.brand_names:
                logger.info(f"=== Procesando marca: {brand_name} ===")
                
                brand_coverages = []
                brand_niq_values = []
                brand_client_values = []
                brand_period_names = []
                brand_trend_niq = 99.9
                brand_trend_client = 99.9
                brand_error = None
                
                try:
                    # Normalizar el nombre de la marca
                    brand_normalized = normalize_string(brand_name)
                    
                    # Filtrar df_niq_copy por marca (columna 0 porque se borraron las columnas no numéricas)
                    niq_col_brand = df_niq_copy.columns[0]
                    df_niq_brand = df_niq_copy[
                        df_niq_copy[niq_col_brand].apply(normalize_string) == brand_normalized
                    ].copy()
                    
                    # Filtrar df_client_copy por marca (columna Homologado_B)
                    idx_brand_client = df_client_copy.columns.get_loc("Homologado_B") if "Homologado_B" in df_client_copy.columns else -1
                    
                    client_col_brand = df_client_copy.columns[idx_brand_client]
                    df_client_brand = df_client_copy[
                        df_client_copy[client_col_brand].apply(normalize_string) == brand_normalized
                    ].copy()
                    
                    logger.info(f"Marca '{brand_name}': NIQ={len(df_niq_brand)} filas, Cliente={len(df_client_brand)} filas")
                    
                    # Validar que hay datos para esta marca
                    if len(df_niq_brand) == 0 or len(df_client_brand) == 0:
                        brand_error = f"No hay datos para la marca '{brand_name}'"
                        logger.warning(brand_error)
                    else:
                        # Obtener columnas numéricas
                        brand_niq_numeric_cols = df_niq_brand.select_dtypes(include=[np.number]).columns.tolist()
                        brand_client_numeric_cols = df_client_brand.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if len(brand_niq_numeric_cols) == 0 or len(brand_client_numeric_cols) == 0:
                            brand_error = f"No hay columnas numéricas para la marca '{brand_name}'"
                            logger.warning(brand_error)
                        else:
                            # Obtener valores de NIQ y Cliente para esta marca
                            brand_niq_row_values = df_niq_brand.iloc[0][brand_niq_numeric_cols].values
                            brand_client_row_values = df_client_brand.iloc[0][brand_client_numeric_cols].values
                            
                            # Calcular cobertura para cada período
                            for i in range(len(brand_niq_numeric_cols)):
                                brand_period_names.append(brand_niq_numeric_cols[i])
                                brand_niq_values.append(float(brand_niq_row_values[i]) if i < len(brand_niq_row_values) else 0)
                                brand_client_values.append(float(brand_client_row_values[i]) if i < len(brand_client_row_values) else 0)
                                
                                # Solo calcular cobertura si hay suficientes datos (1 año)
                                if i >= periods_per_year:
                                    start_idx = i - periods_per_year
                                    end_idx = i
                                    
                                    brand_niq_sum = sum(brand_niq_row_values[start_idx:end_idx+1])
                                    brand_client_sum = sum(brand_client_row_values[start_idx:end_idx+1])
                                    
                                    if brand_client_sum > 0:
                                        brand_coverage = round((brand_niq_sum / brand_client_sum) * 100, 2)
                                    else:
                                        brand_coverage = 0
                                    
                                    brand_coverages.append(brand_coverage)
                                else:
                                    brand_coverages.append(0)
                            
                            # Calcular tendencias
                            if len(brand_niq_values) >= periods_per_year * 2:
                                brand_niq_last_year_sum = sum(brand_niq_values[-periods_per_year:])
                                brand_niq_previous_year_sum = sum(brand_niq_values[-periods_per_year * 2:-periods_per_year])
                                
                                if brand_niq_previous_year_sum > 0:
                                    brand_trend_niq = round(((brand_niq_last_year_sum / brand_niq_previous_year_sum) - 1) * 100, 2)
                                    logger.info(f"Trend NIQ '{brand_name}' calculado: {brand_trend_niq}%")
                            
                            if len(brand_client_values) >= periods_per_year * 2:
                                brand_client_last_year_sum = sum(brand_client_values[-periods_per_year:])
                                brand_client_previous_year_sum = sum(brand_client_values[-periods_per_year * 2:-periods_per_year])
                                
                                if brand_client_previous_year_sum > 0:
                                    brand_trend_client = round(((brand_client_last_year_sum / brand_client_previous_year_sum) - 1) * 100, 2)
                                    logger.info(f"Trend Cliente '{brand_name}' calculado: {brand_trend_client}%")
                            
                            logger.info(f"Métricas calculadas para '{brand_name}': {len(brand_coverages)} períodos")
                    
                except Exception as e:
                    brand_error = f"Error calculando métricas para marca '{brand_name}': {str(e)}"
                    logger.error(brand_error)
                
                # Construir respuesta para esta marca
                brand_response = []
                
                if brand_error:
                    brand_response = [{"error": brand_error}, {"error": brand_error}]
                else:
                    # Scorecard
                    brand_time_frame = brand_period_names[-1] if brand_period_names else "N/A"
                    brand_latest_mat = brand_coverages[-1] if brand_coverages else 0
                    brand_mat_yago = brand_coverages[-1 - periods_per_year] if len(brand_coverages) > periods_per_year else 0
                    brand_difference = round(brand_latest_mat - brand_mat_yago, 2)
                    
                    brand_scorecard = {
                        "manufacturer": manufacturer,
                        "brand": brand_name,
                        "time_frame": brand_time_frame,
                        "mat_yago": brand_mat_yago,
                        "latest_mat": brand_latest_mat,
                        "difference": brand_difference,
                        "trend_niq": brand_trend_niq,
                        "trend_pepsico": brand_trend_client,
                        "wd": wd,
                        "nd": nd,
                        "share": share,
                        "drill_down_level": request.drill_down_level
                    }
                    
                    # Chart
                    brand_chart_data = []
                    for i in range(len(brand_period_names)):
                        brand_chart_data.append({
                            "period": brand_period_names[i],
                            "coverage": brand_coverages[i],
                            "nielseniq": round(brand_niq_values[i], 2),
                            "client": round(brand_client_values[i], 2)
                        })
                    
                    brand_date_range = f"{brand_period_names[0]} - {brand_period_names[-1]}" if brand_period_names else "N/A"
                    
                    brand_chart = {
                        "title": f"{brand_name} Coverage Trends",
                        "description": f"{brand_name} trends for the selected period",
                        "data": brand_chart_data,
                        "metadata": {
                            "max_months": len(brand_period_names),
                            "date_range": brand_date_range
                        }
                    }
                    
                    brand_response = [brand_scorecard, brand_chart]
                
                # Agregar al diccionario de resultados
                results[brand_name] = brand_response
            
            logger.info("Cálculo de cobertura por marcas completado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"Error calculando cobertura por marca: {str(e)}")
            return {"error": f"Error en paso 5 (calcular cobertura): {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error general en calculate_coverage_brands: {str(e)}")
        return {"error": f"Error general: {str(e)}"}


@app.post("/unify-channels-brands")
async def unify_channels_brands(request: UnifyChannelsBrandsRequest):
    """
    Unificar canales y marcas creando columnas homologadas
    
    Parámetros:
    - dicc_niq_channels: Diccionario de canales NIQ {'Modern':[market1,market2],'DTT':[market3,market4]}
    - dicc_client_channels: Diccionario de canales Cliente {'Modern':[market5,market6],'DTT':[market7],'NoNIQ':[market8]}
    - dicc_brand_mapping: Diccionario de mapeo de marcas {Brand1:[Branda,Brandb,Brandc,Brandd]}
    - list_product_deletion: Lista de productos a eliminar [Product1,Product2,Product3]
    - input_data_type: Tipo de dato (Value, Grams, Kilograms, Liters, Milliliters)
    - analysis_periodicity: Periodicidad (monthly, bimonthly)
    - fact_mapping: Diccionario de mapeo de facts {'Sales':'Vtas EQ2','Share':'Part. de Vtas EQ2 - Product'}
    """
    try:
        logger.info("=== Iniciando proceso de unificación de canales y marcas ===")
        
        # Validar que hay datos cargados en el DataManager
        if data_manager.df_client.empty or data_manager.df_niq.empty:
            error_msg = "No hay datos cargados en el DataManager. Primero ejecute validate-excel-files."
            logger.error(error_msg)
            return [{"error": error_msg}]
        
        # Validar que los DataFrames tienen las columnas necesarias
        if len(data_manager.df_client.columns) < 5:
            error_msg = f"df_client debe tener al menos 5 columnas, pero tiene {len(data_manager.df_client.columns)}"
            logger.error(error_msg)
            return [{"error": error_msg}]
        
        if len(data_manager.df_niq.columns) < 2:
            error_msg = f"df_niq debe tener al menos 2 columnas, pero tiene {len(data_manager.df_niq.columns)}"
            logger.error(error_msg)
            return [{"error": error_msg}]
        
        # Los parámetros ya están validados por Pydantic, pero agregamos logs
        logger.info(f"Procesando request con input_data_type: {request.input_data_type}, periodicity: {request.analysis_periodicity}")
        logger.info(f"Canales NIQ: {len(request.dicc_niq_channels)} grupos")
        logger.info(f"Canales Cliente: {len(request.dicc_client_channels)} grupos")
        logger.info(f"Mapeo de marcas: {len(request.dicc_brand_mapping)} marcas principales")
        
        error_messages = []
        
        # 1. Crear columna Homologado_C en df_client basada en columna 3 (CHANNEL)
        try:
            # Crear mapeo inverso para cliente
            client_channel_mapping = {}
            for channel, markets in request.dicc_client_channels.items():
                for market in markets:
                    client_channel_mapping[market] = channel
            
            # Aplicar mapeo a df_client columna 3
            data_manager.df_client['Homologado_C'] = data_manager.df_client.iloc[:, 3].map(client_channel_mapping).fillna('NO NIQ')
            logger.info(f"Columna Homologado_C creada en df_client con {len(data_manager.df_client['Homologado_C'].unique())} valores únicos")
            
        except Exception as e:
            error_msg = f"Error creando columna Homologado_C en df_client: {str(e)}"
            error_messages.append(error_msg)
            logger.error(error_msg)
        
        # 2. Crear columna Homologado_C en df_niq basada en columna 1 (MARKETS)
        try:
            # Crear mapeo inverso para NIQ
            niq_channel_mapping = {}
            for channel, markets in request.dicc_niq_channels.items():
                for market in markets:
                    niq_channel_mapping[market] = channel
            
            # Aplicar mapeo a df_niq columna 1
            data_manager.df_niq['Homologado_C'] = data_manager.df_niq.iloc[:, 1].map(niq_channel_mapping).fillna('NO NIQ')
            logger.info(f"Columna Homologado_C creada en df_niq con {len(data_manager.df_niq['Homologado_C'].unique())} valores únicos")
            
        except Exception as e:
            error_msg = f"Error creando columna Homologado_C en df_niq: {str(e)}"
            error_messages.append(error_msg)
            logger.error(error_msg)
        
        # 3. Crear columna Homologado_B en df_client basada en columna 4 (BRAND)
        try:
            # Crear mapeo inverso para marcas
            brand_mapping = {}
            for brand, sub_brands in request.dicc_brand_mapping.items():
                for sub_brand in sub_brands:
                    brand_mapping[sub_brand] = brand
            
            # Aplicar mapeo a df_client columna 4
            data_manager.df_client['Homologado_B'] = data_manager.df_client.iloc[:, 4].map(brand_mapping).fillna('NO NIQ')
            logger.info(f"Columna Homologado_B creada en df_client con {len(data_manager.df_client['Homologado_B'].unique())} valores únicos")
            
        except Exception as e:
            error_msg = f"Error creando columna Homologado_B en df_client: {str(e)}"
            error_messages.append(error_msg)
            logger.error(error_msg)
        
        # 4. Asignar parámetros al DataManager
        try:
            data_manager.set_processing_params(request.input_data_type, request.analysis_periodicity)
            data_manager.fact_selections = request.fact_mapping
            logger.info(f"Parámetros asignados: input_data_type={request.input_data_type}, periodicity={request.analysis_periodicity}")
            logger.info(f"Fact mapping asignado: {len(request.fact_mapping)} mapeos de facts")
            
        except Exception as e:
            error_msg = f"Error asignando parámetros al DataManager: {str(e)}"
            error_messages.append(error_msg)
            logger.error(error_msg)
        
        # Si hubo errores, retornar lista con errores
        if error_messages:
            return [{"error": "; ".join(error_messages)}]
        
        # 5. Crear lista de diccionarios con value_counts de df_niq
        try:
            result_list = []
            
            # Value counts de Homologado_C en df_niq
            if 'Homologado_C' in data_manager.df_niq.columns:
                homologado_c_counts = data_manager.df_niq['Homologado_C'].value_counts().to_dict()
                result_list.append({
                    "column": "Homologado_C",
                    "source": "df_niq",
                    "value_counts": homologado_c_counts
                })
            
            # Value counts de Homologado_B en df_client
            if 'Homologado_C' in data_manager.df_client.columns:
                homologado_c_counts = data_manager.df_client['Homologado_C'].value_counts().to_dict()
                result_list.append({
                    "column": "Homologado_C", 
                    "source": "df_client",
                    "value_counts": homologado_c_counts
                })

            # Value counts de Homologado_C en df_client
            if 'Homologado_B' in data_manager.df_client.columns:
                homologado_b_counts = data_manager.df_client['Homologado_B'].value_counts().to_dict()
                result_list.append({
                    "column": "Homologado_B", 
                    "source": "df_client",
                    "value_counts": homologado_b_counts
                })
            
            logger.info(f"Proceso completado exitosamente. Retornando {len(result_list)} diccionarios de conteos")
            return result_list
            
        except Exception as e:
            error_msg = f"Error generando value_counts: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg}]
        
    except Exception as e:
        import traceback
        error_msg = f"Error general en unify_channels_brands: {str(e)}"
        logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return [{"error": error_msg, "details": str(e), "type": type(e).__name__}]

@app.post("/validate-excel-files")
async def validate_excel_files(
    cliente_file: UploadFile = File(..., description="Archivo Excel con información del cliente"),
    niq_file: UploadFile = File(..., description="Archivo Excel con datos NIQ")
):
    """
    Validar archivos Excel de Cliente y NIQ según especificaciones de negocio
    
    Input: Dos Excel (cliente y NIQ)
    Output: Un diccionario con validación y tamaños
    """
    try:
        # Validar que ambos archivos sean Excel
        for file, name in [(cliente_file, "cliente"), (niq_file, "NIQ")]:
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"El archivo {name} debe ser un Excel (.xlsx o .xls)"
                )
        

        
        # Leer archivos y convertir a DataFrames
        cliente_contents = await cliente_file.read()
        cliente_df = pd.read_excel(io.BytesIO(cliente_contents))
        
        niq_contents = await niq_file.read()
        niq_df = pd.read_excel(io.BytesIO(niq_contents))
        
        # Verificar información del cliente
        cliente_result = check_excel_cliente(cliente_df)
        logger.info(f"Validación cliente: {cliente_result}")
        
        # Verificar información de NIQ
        niq_result = check_excel_niq(niq_df)
        logger.info(f"Validación NIQ: {niq_result}")
        
        # Extraer información de los resultados [size, mensaje, error, tamanios]
        cliente_size, cliente_mensaje, cliente_error, cliente_tamanios = cliente_result
        niq_size, niq_mensaje, niq_error, niq_tamanios = niq_result
        
        # Verificar si hay errores en alguno de los archivos
        if cliente_error:
            logger.warning(f"Error en cliente: {cliente_mensaje}")
            return {
                "success": False,
                "error": True,
                "message": f"Error en datos del cliente: {cliente_mensaje}",
                "cliente_result": cliente_result,
                "niq_result": niq_result,
                "cliente_filename": cliente_file.filename,
                "niq_filename": niq_file.filename,
                "cliente_shape": cliente_df.shape,
                "niq_shape": niq_df.shape,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
        
        if niq_error:
            logger.warning(f"Error en NIQ: {niq_mensaje}")
            return {
                "success": False,
                "error": True,
                "message": f"Error en datos NIQ: {niq_mensaje}",
                "cliente_result": cliente_result,
                "niq_result": niq_result,
                "cliente_filename": cliente_file.filename,
                "niq_filename": niq_file.filename,
                "cliente_shape": cliente_df.shape,
                "niq_shape": niq_df.shape,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
        
        # Obtener número de columnas numéricas (último elemento de tamanios)
        cliente_numeric_cols = cliente_tamanios[-1] if cliente_tamanios else 0
        niq_numeric_cols = niq_tamanios[-1] if niq_tamanios else 0
        
        # Aplicar regla: niq_numeric_cols >= cliente_numeric_cols
        if niq_numeric_cols < cliente_numeric_cols:
            error_message = f"Error de datos: NIQ tiene {niq_numeric_cols} columnas numéricas, pero cliente tiene {cliente_numeric_cols}. NIQ debe tener >= columnas numéricas que cliente."
            logger.warning(error_message)
            return {
                "success": False,
                "error": True,
                "message": error_message,
                "cliente_result": cliente_result,
                "niq_result": niq_result,
                "cliente_numeric_cols": cliente_numeric_cols,
                "niq_numeric_cols": niq_numeric_cols,
                "cliente_filename": cliente_file.filename,
                "niq_filename": niq_file.filename,
                "cliente_shape": cliente_df.shape,
                "niq_shape": niq_df.shape,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
        
        # Extraer valores únicos de columnas no numéricas
        lista_valores_unicos_niq = []
        lista_valores_unicos_client = []
        list_facts_niq = [] 

        # Extraer valores únicos de las columnas no numéricas del cliente
        # cliente_size contiene el número de columnas no numéricas
        for i in range(cliente_size):
            if i < len(cliente_df.columns):
                valores_unicos = cliente_df.iloc[:, i].dropna().unique().tolist()
                lista_valores_unicos_client.append(valores_unicos)
                logger.info(f"Cliente columna {i} ({cliente_df.columns[i]}): {len(valores_unicos)} valores únicos")
        
        # Extraer valores únicos de las columnas no numéricas de NIQ
        # niq_size contiene el número de columnas no numéricas
        for i in range(niq_size):
            if i < len(niq_df.columns):
                valores_unicos = niq_df.iloc[:, i].dropna().unique().tolist()
                lista_valores_unicos_niq.append(valores_unicos)
                logger.info(f"NIQ columna {i} ({niq_df.columns[i]}): {len(valores_unicos)} valores únicos")
        
        logger.info(f"Total de listas de valores únicos extraídas para NIQ: {len(lista_valores_unicos_niq)} y para Cliente: {len(lista_valores_unicos_client)}")
        
        # Extraer valores únicos de la última columna no numérica de NIQ para list_facts_niq
        try:
            if niq_size > 0 and niq_size <= len(niq_df.columns):
                # La última columna no numérica es la columna en índice (niq_size - 1)
                ultima_columna_index = niq_size - 1
                valores_facts_niq = niq_df.iloc[:, ultima_columna_index].dropna().unique().tolist()
                list_facts_niq = valores_facts_niq
                logger.info(f"Facts NIQ extraídos de columna {ultima_columna_index} ({niq_df.columns[ultima_columna_index]}): {len(list_facts_niq)} valores únicos")
            else:
                logger.warning("No se pudo extraer facts_niq: niq_size inválido o fuera de rango")
                list_facts_niq = []
        except Exception as e:
            logger.error(f"Error extrayendo facts_niq: {str(e)}")
            list_facts_niq = []
        
        # Guardar copia del DataFrame original del cliente (raw, sin NLP)
        cliente_df_raw = cliente_df.copy()
        niq_df_raw = niq_df.copy()
        
        # Aplicar validación NLP si el cliente tiene 7 u 8 columnas no numéricas
        nlp_applied = False
        nlp_rows_deleted = 0
        nlp_message = ""
        
        if cliente_size in [7, 8]:
            logger.info(f"Cliente tiene {cliente_size} columnas no numéricas. Aplicando validación NLP...")
            cliente_df_enriched, nlp_message, nlp_error, nlp_rows_deleted = validate_client_with_nlp(
                cliente_df, 
                cliente_size
            )
            
            if nlp_error:
                logger.error(f"Error en validación NLP: {nlp_message}")
                return { 
                    "success": False,
                    "error": True,
                    "message": f"Error en validación NLP: {nlp_message}",
                    "cliente_filename": cliente_file.filename,
                    "niq_filename": niq_file.filename,
                    "validation_timestamp": pd.Timestamp.now().isoformat()
                }
            
            # Actualizar cliente_df con el DataFrame enriquecido (para df_client)
            cliente_df = cliente_df_enriched
            nlp_applied = True
            logger.info(f"NLP aplicado exitosamente. {nlp_message}")
        else:
            logger.info(f"Cliente tiene {cliente_size} columnas no numéricas. NLP no aplicable (solo para 7 u 8 columnas)")
        
        # Si la validación es exitosa y niq_numeric_cols >= cliente_numeric_cols,
        # asignar los datos al DataManager
        try:
            data_manager.set_data(
                df_client=cliente_df,  # DataFrame procesado (con NLP si aplica)
                df_niq=niq_df,
                df_client_raw=cliente_df_raw,  # DataFrame original sin NLP
                df_niq_raw=niq_df_raw,  # DataFrame original de NIQ
                non_num_client=cliente_size,
                non_num_niq=niq_size,
                drill_down_level="None"  # Valor por defecto
            )
            logger.info("Datos asignados exitosamente al DataManager")
        except Exception as e:
            logger.error(f"Error asignando datos al DataManager: {str(e)}")
            # Continuar con la respuesta exitosa aunque falle la asignación
        
        # Si todo está bien, retornar éxito con los tamaños
        logger.info(f"Validación exitosa - Cliente: {cliente_file.filename}, NIQ: {niq_file.filename}")
        
        response_data = {
            "success": True,
            "error": False,
            "message": "Validación exitosa",
            "cliente_tamanios": cliente_tamanios,
            "niq_tamanios": niq_tamanios,
            "cliente_result": cliente_result,
            "niq_result": niq_result,
            "cliente_numeric_cols": cliente_numeric_cols,
            "niq_numeric_cols": niq_numeric_cols,
            "cliente_filename": cliente_file.filename,
            "niq_filename": niq_file.filename,
            "cliente_shape": cliente_df.shape,
            "niq_shape": niq_df.shape,
            "valores_unicos_niq": lista_valores_unicos_niq,
            "valores_unicos_client": lista_valores_unicos_client,
            "facts_niq": list_facts_niq,
            "data_manager_status": data_manager.get_status(),
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Agregar información de NLP si se aplicó
        if nlp_applied:
            response_data["nlp_applied"] = True
            response_data["nlp_message"] = nlp_message
            response_data["nlp_rows_deleted"] = nlp_rows_deleted
            logger.info(f"Información NLP agregada al response: {nlp_rows_deleted} filas eliminadas")
        else:
            response_data["nlp_applied"] = False
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error validando archivos Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validando archivos: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)