"""
Script mejorado para iniciar el servidor FastAPI con manejo de errores
"""

import uvicorn
import sys
import socket
from pathlib import Path

def check_port_available(host, port):
    """Verificar si el puerto está disponible"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except socket.error:
        return False

def main():
    # Configuración
    host = "127.0.0.1"  # localhost
    port = 8000
    
    print("🚀 Iniciando Excel Validation API...")
    print("=" * 50)
    
    # Verificar que el archivo main.py existe
    if not Path("main.py").exists():
        print("❌ Error: No se encuentra el archivo main.py")
        print("   Asegúrate de estar en el directorio correcto")
        sys.exit(1)
    
    # Verificar que el puerto esté disponible
    if not check_port_available(host, port):
        print(f"⚠️  Advertencia: El puerto {port} ya está en uso")
        print("   Intentando con puerto alternativo...")
        
        # Buscar puerto alternativo
        for alt_port in range(8001, 8010):
            if check_port_available(host, alt_port):
                port = alt_port
                print(f"✅ Usando puerto alternativo: {port}")
                break
        else:
            print("❌ Error: No se encontró un puerto disponible")
            sys.exit(1)
    
    # URLs de acceso
    print(f"🌐 Servidor iniciando en:")
    print(f"   • URL Principal: http://localhost:{port}")
    print(f"   • Documentación: http://localhost:{port}/docs")
    print(f"   • Redoc: http://localhost:{port}/redoc")
    print(f"   • Endpoint: POST http://localhost:{port}/validate-excel-files")
    print("=" * 50)
    print("📝 Presiona Ctrl+C para detener el servidor")
    print()
    
    try:
        # Iniciar servidor
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error iniciando servidor: {e}")
        print("\n🔧 Posibles soluciones:")
        print("   1. Verificar que todas las dependencias están instaladas:")
        print("      pip install -r requirements.txt")
        print("   2. Verificar que no hay errores en el código:")
        print("      python -m py_compile main.py")
        print("   3. Intentar con un puerto diferente")
        sys.exit(1)

if __name__ == "__main__":
    main()

