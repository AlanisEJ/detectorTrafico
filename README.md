# Detector de tráfico — SIT

Base modular para adquirir video RTSP/local, seguir vehículos con YOLO de segmentación + ByteTrack y registrar entradas y permanencias por zona. Python 3.11 o posterior; operación local, incluso en un servidor sin Internet.

Esta reorganización conserva los datos del prototipo de febrero de 2026. El modelo `yolov8m-seg.pt` solicitado por ese prototipo **no está incluido**. La demo y las pruebas funcionan sin él; la captura con inferencia requiere aportar ese modelo y validar el equipo de destino.

## Probar ahora, sin cámara ni GPU

Desde la raíz del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m sit check
python -m sit demo
python -m unittest discover -s tests -v
```

En Linux/macOS: `python3 -m venv .venv`, `source .venv/bin/activate` y los mismos comandos `python`. Para probar directamente desde el código, sin instalar el paquete: `PYTHONPATH=src python -m unittest discover -s tests -v` en bash, o `$env:PYTHONPATH="src"` antes del comando en PowerShell.

La demo produce seis eventos: una visita completa de 4 segundos, una pérdida de seguimiento y un cierre incompleto, además de sus tres entradas. También simula una desaparición breve que no corta la visita. Cada ejecución utiliza una sesión nueva.

`check` valida estructura, zonas y parámetros; **no garantiza disponibilidad de cámara/modelo**. `check --runtime` comprueba archivos locales, variables y presencia de dependencias, sin conectar ni cargar el modelo. Tampoco sustituye una prueba de inferencia.

## Ejecutar con una cámara

```powershell
python -m pip install -e ".[vision]"
Copy-Item config/cameras.example.json config/cameras.local.json
$env:SIT_CAM_1_RTSP = 'rtsp://USUARIO:CONTRASENA@HOST/RUTA'
python -m sit model-info artifacts/models/yolov8m-seg.pt
python -m sit check --config config/cameras.local.json --runtime
python -m sit run --config config/cameras.local.json --camera cam_1
```

Antes de esos dos últimos comandos, colocar una copia local del modelo y registrar su checksum en `model.sha256`; ver [modelos](artifacts/models/README.md). `sha256: null` permite preparar la configuración, pero no comprueba integridad. El programa exige una ruta existente y no intenta descargar pesos ni instalar dependencias automáticamente. Rechaza modelos de detección por cajas donde se requieren máscaras.

La URL se lee del entorno y no se guarda en JSON ni se imprime en los logs del programa. `.env.example` es una plantilla; no hay carga automática de `.env`. En bash: `export SIT_CAM_1_RTSP='rtsp://USUARIO:CONTRASENA@HOST/RUTA'`. Codificar caracteres reservados del usuario/contraseña en la URL.

Las dependencias directas de visión están fijadas en `pyproject.toml`; sus dependencias transitivas no forman un lock completo. PyTorch/CUDA deben instalarse con los paquetes apropiados para el controlador y sistema de destino; la selección de CPU/GPU es `model.device` (`auto`, `cpu` o índice CUDA). FP16 solamente se habilita en GPU. No se garantiza rendimiento ni compatibilidad CUDA sin probar el hardware.

Para desplegar sin Internet, preparar el paquete, pesos y un wheelhouse en un equipo con el mismo sistema operativo, arquitectura y versión de Python del destino. Instalar allí con `pip install --no-index --find-links wheelhouse .[vision]` incluyendo las dependencias de construcción. Después de validar inferencia, conservar un `pip freeze --all` del entorno y los checksums junto con el despliegue. No se requiere acceso al servidor para ejecutar las pruebas locales.

Ctrl+C/SIGTERM solicita cierre ordenado. `--max-frames 100` limita una prueba por cámara. Una falla de inferencia o escritura detiene la ejecución con estado de error; una caída RTSP cierra la sesión como incompleta y activa reintentos con espera y timeouts. Los errores detallados de FFmpeg/OpenCV se silencian para evitar exposición de URLs.

## Organización y responsabilidades

| Ruta | Función |
| --- | --- |
| `src/sit/main.py`, `runner.py` | CLI, coordinación y un consumidor de inferencia por cámara |
| `src/sit/capture/rtsp.py` | Captura, reconexión, reloj, secuencia y cola acotada |
| `src/sit/perception/` | Adaptador público de Ultralytics; YOLO/ByteTrack por cámara |
| `src/sit/domain/` | Geometría, ocupación y máquina de estados de visitas |
| `src/sit/storage/` | CSV de eventos y logs operativos diarios |
| `src/sit/ui/` | Editor opcional de polígonos con OpenCV |
| `config/` | Cámaras, tracker y zonas JSON con resolución/cámara explícitas |
| `artifacts/models/` | Pesos locales y manifiesto SHA-256 |
| `data/legacy/`, `config/zones/legacy/`, `legacy/` | Evidencia original y referencia del prototipo |
| `tests/` | Pruebas sin servicios, modelos ni librerías de visión |

`python pruebaVideo.py ...` conserva el acceso al CLI después de instalar el paquete. El monolito se conserva en `legacy/pruebaVideo.py`, con la credencial retirada; no se utiliza como backend del programa nuevo. Importar los módulos de `sit` no abre ventanas, cámaras ni modelos.

Cada cámara tiene lector, cola de dos mensajes, modelo, tracker y estado de eventos propios. RTSP descarta frames antiguos cuando la inferencia se retrasa; los videos locales aplican contrapresión y conservan los frames seleccionados. Cada resultado se consume una sola vez. Una nueva conexión crea una sesión y un tracker nuevos, incluso si un aviso intermedio de desconexión fue desplazado de la cola. Agregar cámaras implica agregar modelos en memoria; todavía no existe un planificador compartido de GPU.

El Kalman adicional del prototipo suavizaba el dibujo. Se omite en la captura sin interfaz: ByteTrack conserva su seguimiento y no se introduce otra asociación de IDs. El suavizado visual podría añadirse como adaptador independiente.

## Configuración, video y zonas

Las rutas dentro del JSON son relativas al **archivo de configuración**, no al directorio desde el cual se ejecuta Python. Los IDs de cámara aceptan letras, números, guion y guion bajo. Las zonas deben coincidir con el ID y la resolución de su cámara; se rechazan polígonos degenerados, cruzados o fuera de límites.

Para probar un video, habilitar `cam_2`, deshabilitar `cam_1` o seleccionar `--camera cam_2`, y colocar el archivo en `data/input/prueba.mp4`. Su `source` tiene `type: "video"` y `path`. Opcionalmente admite `fps` y `start_at` ISO 8601 con zona horaria. Las duraciones usan el tiempo del video, no la velocidad de inferencia; sin `start_at`, la fecha de referencia es el inicio del procesamiento y no la fecha real de grabación. RTSP utiliza reloj monótono para duración y UTC para los timestamps.

```powershell
python -m sit zones --config config/cameras.local.json --camera cam_1
```

El editor trabaja sobre la primera zona o la seleccionada con `--zone ID`. Clic agrega un vértice; arrastrar mueve uno existente. **S** valida y guarda; **L** vuelve a cargar; **C** limpia; **ESC** sale. La captura debe reiniciarse para adoptar cambios. Requiere entorno gráfico; `run` funciona sin escritorio. Los JSON migrados conservan los puntos originales a 1280×720; su correspondencia con la imagen actual de cada cámara requiere revisión en campo.

## Eventos y almacenamiento

| Evento | Significado | `duration_s` | `complete` |
| --- | --- | --- | --- |
| `ENTRADA_ZONA` | Primera observación del centro de máscara dentro del polígono, incluido su borde | 0 | false |
| `ZONA_TRAFICO` | Observaciones fuera de zona durante la confirmación configurada | Desde entrada hasta primera observación exterior | true |
| `SEGUIMIENTO_PERDIDO` | Ausencia de observaciones del ID mayor que `lost_timeout_s` | Vacío | false |
| `SESION_CERRADA` | Visita abierta al desconectar, detener, terminar video o fallar | Vacío | false |

`exit_confirm_s=0.5` requiere volver a observar el mismo ID fuera; un solo resultado vacío no confirma una salida. Una recuperación dentro de `lost_timeout_s=5` conserva la visita. Una pérdida prolongada seguida por el mismo ID abre otra visita. El nombre `ENTRADA_ZONA` no equivale a cruce de una línea física de entrada: esa regla del SIT queda pendiente.

La ocupación observada y su nivel (BAJO ≤3, MEDIO ≤8, ALTO >8) se reportan periódicamente en diagnóstico. Representan vehículos observados dentro de zonas, no flujo por intervalo ni una identidad física garantizada entre cámaras/sesiones.

Los datos se guardan en `data/YYYY-MM-DD/database/CAMERA_ID/trafico.csv` y los diagnósticos en `data/YYYY-MM-DD/logs/trafico.log`. El día se calcula en la zona horaria configurada, `America/Hermosillo` en el ejemplo. Los timestamps de evento son UTC. Los CSV nuevos nunca se concatenan con los logs históricos.

| Campos CSV v1 | Contenido |
| --- | --- |
| `schema_version`, `event_id` | Versión y clave de evento para deduplicación posterior |
| `session_id`, `visit_id` | Ejecución del tracker y visita a una zona |
| `camera_id`, `zone_id`, `track_id` | Origen, polígono e ID local del tracker |
| `class_id`, `vehicle_type` | Clase COCO: 2 Auto, 3 Moto, 5 Bus, 7 Camion |
| `event_type`, `timestamp_utc`, `frame_id` | Evento, instante y secuencia de captura |
| `duration_s`, `observed_duration_s` | Duración completa o vacío; intervalo observado dentro, sin extrapolación |
| `complete`, `reason` | Completitud y motivo del cierre |

Un único proceso coordina la escritura de sus cámaras. No ejecutar dos procesos contra el mismo directorio de salida: no hay bloqueo entre procesos, journal transaccional ni recuperación automática tras corte eléctrico. Los eventos ya escritos se cierran por fila; una terminación forzada puede dejar visitas sin cierre. La rotación es diaria, sin eliminación automática del historial.

Para Pandas/Streamlit, leer el esquema nuevo y filtrar `event_type == "ZONA_TRAFICO"` y `complete == True` antes de calcular permanencia media. Filtrar por duración y agrupar por cámara/clase/intervalo es una etapa analítica separada. El CSV v1 no reemplaza silenciosamente `analisis_trafico_final.csv` ni el esquema anterior de seis columnas.

## Validación y siguientes hitos

Las pruebas cubren pérdidas temporales/prolongadas, reentradas, confirmación de salida, aislamiento de cámaras/sesiones, frames repetidos, relojes de video, colas y cierre, validación de zonas/modelos, concurrencia del CSV y CLI. El adaptador YOLO y el backend OpenCV se prueban con dobles; eso no certifica inferencia real.

Pendiente en el entorno local de destino: aportar el modelo de segmentación, verificar checksum, validar CPU/CUDA/FP16 con un video conocido, calibrar zonas y confirmar la reconexión con la cámara real. El administrador debe rotar las credenciales publicadas previamente; este cambio no accede al dispositivo ni reescribe el historial del repositorio.

Después: cruce direccional de línea `ENTRADA`, normalización histórica separada, agregados Pandas compatibles con el dashboard del SIT, medición de rendimiento multicámara y planificador GPU si hace falta. La predicción de tráfico y el dashboard no forman parte de esta base.

Referencias primarias: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [versión fijada](https://pypi.org/project/ultralytics/8.3.221/) y [paquetes PyTorch por plataforma](https://pytorch.org/get-started/previous-versions/).
