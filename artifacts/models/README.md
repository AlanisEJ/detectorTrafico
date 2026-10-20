# Modelos locales

El script original solicitaba `yolov8m-seg.pt`, pero ese archivo nunca estuvo en el repositorio. No se ha descargado ni sustituido: `sit run` falla antes de conectar si falta.

Los dos `.pt` existentes se conservan aquí con el mismo contenido. Sus nombres corresponden a modelos de detección por cajas; no se han deserializado para ejecutar esta reorganización. El adaptador comprueba `model.task == "segment"` y rechaza modelos sin máscaras.

Para habilitar inferencia:

1. Preparar en un equipo con conexión una copia del modelo de segmentación autorizado para el SIT.
2. Transportarla al servidor sin Internet y colocarla aquí, o indicar otra ruta local en la configuración.
3. Calcular `python -m sit model-info artifacts/models/yolov8m-seg.pt` y registrar el SHA-256 en la configuración y el manifiesto, junto con origen y versión.
4. Ejecutar `python -m sit check --config config/cameras.local.json --runtime`.

Los pesos nuevos se excluyen de Git ordinario. Distribuirlos como artefactos versionados con checksum. Los dos pesos históricos siguen rastreados para conservarlos; migrar su historial a LFS queda fuera de este cambio.
