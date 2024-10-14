# Detección de Huecos - Transformation Day

Usando algoritmo YOLO para detección de objetos en tiempo real, las cámaras de los carros detectan huecos en la vía para implementar un sistema de suspensión adaptativa de acuerdo al estado de la carretera

* Se pueden obtener mejores resultados usando yolov8 - nano, pensado a implementar en un microprocesador a futuro.
* Imágenes indicativas de izquierda / derecha, señalan el lugar donde se predicen los huecos para activar dicho lado de la suspensión y cierta proporción.
* Se puede comunicar python con Arduino, para ver una breve implementación física.
