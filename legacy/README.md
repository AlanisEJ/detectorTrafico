# Referencia del prototipo

`pruebaVideo.py` conserva el código del commit `eddbd81`, excepto la URL RTSP: se sustituyó por la variable `SIT_CAM_1_RTSP` para no volver a publicar credenciales. No es el punto de entrada recomendado ni debe importarse en las pruebas.

Los logs originales están en `data/legacy/`; los tres `.npy` originales en `config/zones/legacy/`. Su contenido se conserva sin modificar. La conversión a JSON se hizo con `numpy.load(..., allow_pickle=False)` y mantiene las coordenadas originales a 1280×720.

Las credenciales publicadas anteriormente requieren rotación en el dispositivo/servidor por su administrador. Retirarlas de la rama nueva no las invalida ni elimina el historial público.
