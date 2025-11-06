"""
Script para iniciar el servidor FastAPI
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 Iniciando servidor FastAPI...")
    print("📋 Excel Validation API")
    print("🔗 URL: http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs")
    print("🔧 Redoc: http://localhost:8000/redoc")
    print("✅ Endpoint: POST /validate-excel-files")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Cambiar a localhost para evitar problemas de conectividad
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )

