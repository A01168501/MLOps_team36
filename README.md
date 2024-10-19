# TEAM 36

- **Héctor Antonio Tafoya Garcia** - A00466155
- **Gerardo Quiroga Najera** - A00967999
- **David Nava Jiménez** - A01168501
- **Fernando Jimenez Pereyra** - A01734609
- **Donnovan Alfredo Ramírez Rodríguez** - A01795652

## FASE 1
Documentar y generar estructura inicial del proyecto

## FASE 2
Configurar Docker y MLflow


## Como ejecutar el proyecto
Luego de clonar el proyecto, posicionar la terminal en la carpeta raiz, posteriormente:

**Activar el Virtual Environment:**

   - **On Windows:**

     ```bash
     .\itesm_venv\Scripts\activate
     ```

     or if using Git Bash

     ```bash
     source \itesm_venv\Scripts\activate
     ```

   - **On Linux and MacOS:**

     ```bash
     source itesm_venv/bin/activate


**Construir la imagen:**
docker-compose build

**Iniciar los contenedores**
docker-compose up

**Confirmar mlflow instalado correctamente**
Puede consultarlo accediendo a http://localhost:5000



## Problema: Air Quality

El dataset con el que vamos a trabajar corresponde a “air quality”, el cual es un dataset multivariable con el que se pueden aplicar técnicas de series de tiempo. El objetivo es obtener por medio de regresión la predicción de la variable objetivo. Cuenta con 9,358 instancias u observaciones y 15 variables distintas.

A continuación, se describe la información del dataset que viene detallado en (Vito, 2016):

“El dataset contiene 9,358 instancias de respuestas promedio por hora de una matriz de 5 sensores químicos de óxido metálico integrados en un dispositivo multisensor de calidad del aire. El dispositivo se ubicó en el campo en una zona significativamente contaminada, a nivel de la carretera, dentro de una ciudad italiana. Los datos se registraron desde marzo de 2004 hasta febrero de 2005 (un año), representando las grabaciones más largas disponibles gratuitamente de las respuestas de dispositivos de sensores químicos de calidad del aire desplegados en el campo. Las concentraciones promediadas por hora de referencia para CO, Hidrocarburos No Metánicos, Benceno, Óxidos Totales de Nitrógeno (NOx) y Dióxido de Nitrógeno (NO2) fueron proporcionadas por un analizador certificado de referencia co-ubicado. Los valores faltantes están etiquetados con el valor -200.”

### Información de las Variables

1. **Fecha** (DD/MM/YYYY)
2. **Tiempo** (HH.MM.SS)
3. **Concentración verdadera promediada por hora de CO** en mg/m^3 (analizador de referencia)
4. **Respuesta del sensor promediada por hora PT08.S1** (óxido de estaño) (nominalmente dirigido a CO)
5. **Concentración verdadera promediada por hora de Hidrocarburos No Metánicos** en microg/m^3 (analizador de referencia)
6. **Concentración verdadera promediada por hora de Benceno** en microg/m^3 (analizador de referencia)
7. **Respuesta del sensor promediada por hora PT08.S2** (titania) (nominalmente dirigido a NMHC)
8. **Concentración verdadera promediada por hora de NOx** en ppb (analizador de referencia)
9. **Respuesta del sensor promediada por hora PT08.S3** (óxido de tungsteno) (nominalmente dirigido a NOx)
10. **Concentración verdadera promediada por hora de NO2** en microg/m^3 (analizador de referencia)
11. **Respuesta del sensor promediada por hora PT08.S4** (óxido de tungsteno) (nominalmente dirigido a NO2)
12. **Respuesta del sensor promediada por hora PT08.S5** (óxido de indio) (nominalmente dirigido a O3)
13. **Temperatura** en °C
14. **Humedad Relativa** (%)
15. **AH Humedad Absoluta**

La variable **3 - Concentración verdadera promediada por hora de CO en mg/m^3** (analizador de referencia) será la variable que se va a predecir en el modelo.

## Referencias

Vito, S. (2016, July 21). UC Irvine Machine Learning Repository. Retrieved September 30, 2024, from UC Irvine Machine Learning Repository - Air Quality: [https://archive.ics.uci.edu/dataset/387/air+quality](https://archive.ics.uci.edu/dataset/387/air+quality)


# Predicción de Contaminación de CO

| **TAREA DE PREDICCIÓN**                                                                                             | **DECISIONES**                                                                                                                                       | **PROPUESTA DE VALOR**                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Objetivo:** Predecir las concentraciones de contaminantes en el aire CO (en mg/m³) utilizando técnicas de regresión a partir de datos de sensores químicos y variables ambientales. | Decisiones basadas en predicciones: <ul><li>Implementar políticas de control de la contaminación.</li><li>Mejorar la planificación urbana y del tráfico en áreas contaminadas.</li><li>Informar a la población sobre los niveles de calidad de aire, actividades recomendadas y riesgos asociados.</li></ul> | <ul><li>Proporcionar información precisa sobre la calidad del aire con un enfoque en el CO.</li><li>Contribuir a la salud pública al reducir la exposición de contaminantes.</li><li>Facilitar la investigación sobre el impacto de la contaminación en la salud, medio ambiente y planeación urbana.</li></ul> |

| **RECOLECCIÓN DE DATOS**                                                                                                                                                                                                                   | **FUENTE DE DATOS**                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Método de recolección:** Datos recogidos por un dispositivo multisensor de diferentes compuestos químicos, ubicado en ciudad contaminada de Italia, a nivel de la carretera.<br>**Periodo de recolección:** Desde marzo de 2024 hasta febrero de 2005. | <ul><li>Matriz de 5 sensores químicos integrados en un dispositivo.</li><li>Datos de referencia proporcionados por un analizador certificado co-ubicado.</li><li>9,358 instancias promedio de mediciones por hora.</li></ul> |

| **SIMULACIÓN DE IMPACTO**                                                                                                                                                            | **REALIZACIÓN DE PREDICCIONES**                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Evaluar cómo cambiarían las concentraciones de contaminantes bajo diferentes escenarios de tráfico y producción industrial, o incluso durante eventos esporádicos como incendios.<br>Proyecciones sobre la mejora de la calidad del aire con la implementación de políticas de mitigación. | Aplicar el modelo entrenado a nuevos datos de calidad del aire para estimar las concentraciones futuras de CO.<br>Probar en otras localidades y comparar resultados. |

| **CONSTRUCCIÓN DE MODELOS**                                                                                      | **CARACTERÍSTICAS**                                                                                                                                                                                                                                                                                                                                                                                   |
|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Utilizar técnicas de regresión para modelar relaciones entre las características y la concentración de CO. | <ul><li>**Variable objetivo:** Concentraciones verdadera promediadas de CO</li><li>**Respuestas de sensores:** CO, Hidrocarburos no metálicos, benceno, NOx, NO2.</li><li>**Fecha:** Tendencias estacionales, día de la semana o mes.</li><li>**Tiempo:** Mañana, tarde o noche u hora específica.</li><li>Temperatura</li><li>Humedad Relativa</li><li>Humedad Absoluta</li></ul> |

| **MONITOREO Y CONTROL**                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------|
| <ul><li>Evaluar el rendimiento del modelo regularmente con nuevos datos.</li><li>Ajustar el modelo según sea necesario para mantener su precisión y relevancia.</li></ul> |