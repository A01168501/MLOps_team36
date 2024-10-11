# TEAM 36

- **Héctor Antonio Tafoya Garcia** - A00466155
- **Gerardo Quiroga Najera** - A00967999
- **David Nava Jiménez** - A01168501
- **Fernando Jimenez Pereyra** - A01734609
- **Donnovan Alfredo Ramírez Rodríguez** - A01795652

## FASE 1
Documentar y generar estructura inicial del proyecto

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
