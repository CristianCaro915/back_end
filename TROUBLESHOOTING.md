# 🔧 Guía de Solución de Problemas

## 🚨 Problema: "Parece que la página web podría estar teniendo problemas"

### ¿Qué significa este error?

Este error indica que tu navegador no puede conectarse al servidor FastAPI. Puede ocurrir por varias razones:

1. **El servidor no está ejecutándose**
2. **Problemas de configuración de host/puerto**
3. **Firewall o antivirus bloqueando la conexión**
4. **Conflictos de puerto**

### ✅ Soluciones Paso a Paso

#### 1. Verificar que el servidor esté ejecutándose

```bash
# Activar entorno virtual
.env\Scripts\activate  # Windows
# o
source .env/bin/activate  # Linux/Mac

# Iniciar servidor (opción recomendada)
python run_server.py

# O usar el script original
python start.py

# O directamente con uvicorn
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### 2. URLs Correctas para Acceder

- ✅ **Correcta**: `http://localhost:8000/docs`
- ✅ **Correcta**: `http://127.0.0.1:8000/docs`
- ❌ **Incorrecta**: `http://0.0.0.0:8000/docs`

#### 3. Verificar que el Puerto esté Libre

```bash
# Windows
netstat -an | findstr :8000

# Linux/Mac
lsof -i :8000
```

Si el puerto está ocupado, el script `run_server.py` buscará automáticamente uno alternativo.

#### 4. Probar Conectividad Básica

Antes de ir a `/docs`, prueba primero:
- `http://localhost:8000/` (página principal)
- `http://localhost:8000/health` (health check)

### 🔍 Diagnóstico de Problemas

#### El servidor inicia pero no puedo acceder

```bash
# Verificar si FastAPI está respondiendo
curl http://localhost:8000/health

# O en PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/health"
```

#### Error de dependencias

```bash
# Reinstalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import fastapi, uvicorn, pandas; print('✅ OK')"
```

#### Error de importación

```bash
# Verificar sintaxis
python -m py_compile main.py operations.py routes.py

# Ejecutar directamente
python main.py
```

### 🛠️ Configuraciones Alternativas

#### Opción 1: Servidor Simple (Solo API)
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

#### Opción 2: Con Auto-reload
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### Opción 3: Puerto Alternativo
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

### 🔧 Logs y Debugging

#### Ver logs detallados
```bash
python run_server.py
# Los logs aparecerán en la consola
```

#### Modo debug
```python
# En main.py, agregar al final:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
```

### 🌐 Problemas de Red

#### Firewall Windows
1. Windows + R → `firewall.cpl`
2. "Permitir una aplicación a través de Firewall de Windows"
3. Agregar Python/uvicorn si no está

#### Antivirus
Algunos antivirus bloquean servidores locales. Agregar excepción para:
- La carpeta del proyecto
- Python.exe
- Puerto 8000

### 📱 Probar desde Dispositivos Móviles

Si quieres acceder desde otros dispositivos en la misma red:

```python
# Cambiar host en run_server.py o start.py
host = "0.0.0.0"  # Permitir conexiones externas
```

Luego acceder con la IP de tu computadora:
`http://192.168.1.XX:8000/docs`

### 🆘 Si Nada Funciona

1. **Reinicia el terminal/cmd**
2. **Desactiva temporalmente antivirus/firewall**
3. **Prueba otro navegador**
4. **Verifica que tienes permisos de administrador**
5. **Reinicia la computadora**

### 📞 Comandos de Emergencia

```bash
# Matar todos los procesos de Python
taskkill /F /IM python.exe  # Windows
pkill python  # Linux/Mac

# Limpiar puerto 8000
netsh int ipv4 set global autotuninglevel=disabled  # Windows (requiere admin)

# Probar conectividad básica
ping localhost
telnet localhost 8000
```

### ✅ Verificación Final

Una vez que el servidor esté funcionando, deberías ver:

1. **En la consola**: Mensajes de uvicorn indicando que el servidor está ejecutándose
2. **En http://localhost:8000**: Página JSON con información de la API
3. **En http://localhost:8000/docs**: Interfaz Swagger/OpenAPI
4. **En http://localhost:8000/health**: Status "healthy"

¡Con esto deberías poder resolver cualquier problema de conectividad!

