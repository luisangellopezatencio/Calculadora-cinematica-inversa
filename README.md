# Calculadora-cinematica-inversa



Herramienta desarrollada capaz de calcular la cinemática inversa de robots antropomórficos de seis grados de libertad en un tiempo relativamente corto y con una precisión aceptable, además el modelo obtenido es muy eficiente lo que lo hace ideal para ser usado en aplicaciones en tiempo real, en segundo lugar, desde un punto de vista, de ahorro energético el modelo obtenido es considerablemente mejor, dado que por la forma en que se generó el conjunto de datos para entrenar el modelo este es capaz de arrojar una variación en las variables articulares mucho menor que el modelo de la cinemática inversa calculado de manera tradicional.

En consecuencia, el resultado será una herramienta valiosa para aquellos que trabajen con este tipo de robots y que no tengan un conocimiento profundo de esta área, además este trabajo podría tener un impacto significativo en le campo de la robótica al incrementar de manera considerable la productividad y la eficiencia en el cálculo de la cinemática inversa en robots antropomórficos de seis grados de libertad.

## Dependencias
scikit-learn, pytorch, numpy, matplotlib y tkinter

# Uso de la herramienta

Como ejemplo practico se usará el robot ABB-IRB-2400.
Cuya cinematica directa es la siguiente:
![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/1ee05ba3-9684-4290-8656-43ed4602ca81)

Imágenes extraídas de: Barrios, A., Peñin, L. F., Ballaguer, C., y Aracil, R. (2016). Fundamentos de robótica (2.ª ed.). Madrid: McGraw-Hill.
![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/e5c970bb-2a0f-4088-a02e-cf4601d3f9f6)



[(Back to top)](#table-of-contents)

### Pasos

- `1` : Ingresa los datos que pide la interfáz gráfica (Parametros Denavit-Hartenberg y rango de moviemiento de las articulaciones) separados por coma y sin espacios. reemplaza theta o q por ceros.
![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/7a0f0aa2-fcb1-44af-9494-0a2f7a517dbe)

  

- `2` : Cuando hayas ingresados los datos, dar click en el botón 'Iniciar', esto calculará el espacio de trabajo del robot y graficará el mismo en alambres, también prepará los datos para entrenar el modelo

  ![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/ab1e56c8-efc9-4b92-abd9-0497b6c54657)
  ![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/20ed2938-03bd-4696-8193-525cd874319f)

- `3` : Cuando haya terminado de calcular el espacio de trabajo del robot, dar click en el botón "Calcular". Esto entrenará el modelo y guardará el modelo, en la carpeta raíz.
 ![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/ff7e6d88-5cdc-4b6b-9be9-708eadd36f8c)
 ![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/aa9eac0a-a8fe-4610-b9eb-10ae96b7b377)
  
### Trayectorias.
A continuación se muestra una trayectoria en la cual se aprecia la precisión del modelo

![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/8cc4ce42-088b-494e-90eb-bb3e461d2a45)

https://drive.google.com/file/d/1y4bzqtAlhZWHPrln3-h5ljm5t9bGIyHg/view?usp=drive_link

## Video de Demostración

[![Video de Demostración](![image](https://github.com/luisangellopezatencio/Calculadora-cinematica-inversa/assets/111664276/76cd1ae8-6ccc-4882-9991-fe6bab95e2bf)
)](https://drive.google.com/file/d/1y4bzqtAlhZWHPrln3-h5ljm5t9bGIyHg/view?usp=drive_link)











The MIT License (MIT) 2017
