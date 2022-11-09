Dataset disponible en:
https://drive.google.com/drive/u/0/folders/1EKrtCoFGRzOiy8fkUxnN-pGNEaUXYAxB


*QMUL-TinyFace disponible en:
https://qmul-tinyface.github.io

Código para generar TinyFace-Lanczos a partir de TinyFace disponible en carpeta de Códigos.

**CASIA-Webface (y otros datasets utilizados por Arcface) disponible en el repositorio de Arcface:
https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

Cómo descargar desde Pan-Baidu:
https://gist.github.com/imneonizer/1e55dfd86274a3bb1de48435b52c5b12

***Código para generar CASIA-BL y CASIA-DEG a partir de CASIA-Webface en carpeta de Códigos.

****Loreto's dataset disponible en: ...

Construcción del dataset:

DataTrain:

1. 500 primeras  (N0005136 a N0072715) clases de Casia-Bl_train. 30.942 fotos.
2. 500 segundas  (N0073678 a N0191688) clases de Casia-Deg_train. 23.794 fotos.
3. 1000 primeras (n000002 a n001061) clases de Loreto gen_Bicubic_28x28. 20.047 fotos.
4. 1000 segundas (n001062 a n002135) clases de Loreto gen_Nearest_28x28. 20.268 fotos.
5. 1000 terceras (n002135 a n003221) clases de Loreto gen_Area_56x56. 18.940 fotos.
6. todas las 2.570 clases de TinyFace-Lanczos-Train (N3 a N5697). 7.804 fotos.


DataVal:

1. 50 primeras (N0000045 a N0000240) clases de Casia-Bl_val. 9.495 fotos.    (casi 1/3 del train)
2. 50 últimas (N0005026 a N0005134) clases de Casia-Deg_val. 4.505 fotos.   (casi 1/5 del train)
3. 100 últimas (n009170 a n009279) clases de Loreto gen_Bicubic_28x28. 1.975 fotos.      (casi 1/10 del train)
4. 100 penúltimas (n009066 a n009169) clases de Loreto gen_Nearest_28x28. 1.963 fotos.   (casi 1/10 del train)
5. 100 ante penúltimas (n008963 a n009065) clases de Loreto gen_Area_56x56. 1.801 fotos. (casi 1/10 del train)
6. 1.285 primeras (N1 a N3119) clases de TinyFace-Lanczos_test 4.408 fotos. 


DataTest:

1. 100 primeras (N3981330 a N4292045) clases de Casia-Bl_test. 2903 fotos           (casi 1/10 del train)
2. 100 segundas (N4296357 a N4553772) clases de Casia-Deg_test. 2964 fotos.        (casi 1/8 del train)
3. Me ubico en la carpeta 4310 y cuento 100 (n004625 a n004739) de Loreto_gen_Area_56x56. 2006 fotos.    (casi 1/10 del train)
4. Me ubico en la carpeta 4411 y cuento 100 (n004740 a n004855) de Loreto_gen_Bicubic_28x28. 1975 fotos. (casi 1/10 del train)
5. Me ubico en la carpeta 4512 y cuento 100 (n004856 a n004964) de Loreto_gen_Nearest_28x28. 1975 fotos. (casi 1/10 del train)
6. 1.284 segundas clases de TinyFace-Lanczos_test (N3122 a N5700) 3.763 fotos.

