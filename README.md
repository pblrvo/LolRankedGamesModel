# APRENDIZAJE AUTOMÁTICO LolRankedGamesModel

##### Realizaremos el aprendizaje cogiendo dátos de partidas del videojuego League Of Legends sobre quien gana la partida si el equipo rojo o el equipo azul básandonos en los datos de los 10 primeros minutos de partida. En específico tomaremos los siguientes datos de ambos equipos:
- [ ] Primera kill
- [ ] Número de "Mounstros Élite" eliminados
- [ ] Número de dragones conseguidos
- [ ] Oro total
- [ ] Experiencia Total
- [ ] Diferencia de oro
- [ ] Minions eliminados por minuto
- [ ] Gana el equipo AZUL
##### Dentro de weka los datos quedarían así para realizar las diferentes comprobaciones

```shell
=== Run information ===

Scheme:       weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1
Relation:     high_diamond_ranked_10min-weka.filters.unsupervised.attribute.Reorder-R2,3,4,5,6,7,8,9,10,11,12,13,14,15,1
Instances:    9879
Attributes:   15
              blueFirstBlood
              blueEliteMonsters
              blueDragons
              blueTotalGold
              blueTotalExperience
              blueGoldDiff
              blueCSPerMin
              redFirstBlood
              redEliteMonsters
              redDragons
              redTotalGold
              redTotalExperience
              redGoldDiff
              redCSPerMin
              blueWins
Test mode:    10-fold cross-validation
```

## SOLUCIONES DISTINTOS ALGORITMOS
#### Multilayer Perceptron

```shell

=== Classifier model (full training set) ===

Sigmoid Node 0
    Inputs    Weights
    Threshold    -0.20626935718259112
    Node 2    0.6754256664027997
    Node 3    -0.8450093104064836
    Node 4    -0.6368432850607662
    Node 5    -0.6252960933625291
    Node 6    -1.0970169417384408
    Node 7    1.0426324677655412
    Node 8    0.8643699995362139
    Node 9    4.160813349585542
Sigmoid Node 1
    Inputs    Weights
    Threshold    0.20626935718259107
    Node 2    -0.6754256664027998
    Node 3    0.8450093104064834
    Node 4    0.636843285060767
    Node 5    0.6252960933625291
    Node 6    1.0970169417384403
    Node 7    -1.042632467765541
    Node 8    -0.8643699995362137
    Node 9    -4.160813349585545
Sigmoid Node 2
    Inputs    Weights
    Threshold    -12.33552687700015
    Attrib blueFirstBlood=FALSE    -1.234458362409553
    Attrib blueEliteMonsters    -6.3842574261435345
    Attrib blueDragons=TRUE    2.2361106448618027
    Attrib blueTotalGold    -7.379283436295732
    Attrib blueTotalExperience    -0.4236024780459842
    Attrib blueGoldDiff    -6.270811335463747
    Attrib blueCSPerMin    -7.7852117276282895
    Attrib redFirstBlood=TRUE    -1.1468316855818017
    Attrib redEliteMonsters    -9.407773460901216
    Attrib redDragons=TRUE    5.720274617125802
    Attrib redTotalGold    4.031386802407705
    Attrib redTotalExperience    9.330695507486695
    Attrib redGoldDiff    6.277840388322294
    Attrib redCSPerMin    -4.507749519178261
Sigmoid Node 3
    Inputs    Weights
    Threshold    -11.23168516245016
    Attrib blueFirstBlood=FALSE    -0.9265762399291405
    Attrib blueEliteMonsters    -3.3125649608856405
    Attrib blueDragons=TRUE    2.7543408228757977
    Attrib blueTotalGold    7.366546826694323
    Attrib blueTotalExperience    21.200234431128713
    Attrib blueGoldDiff    13.00603713789047
    Attrib blueCSPerMin    -5.2543966269276146
    Attrib redFirstBlood=TRUE    -0.9372273883338456
    Attrib redEliteMonsters    -2.4029095957106796
    Attrib redDragons=TRUE    0.2931363082600411
    Attrib redTotalGold    -16.60152700262533
    Attrib redTotalExperience    -6.461865682318424
    Attrib redGoldDiff    -12.975430672627681
    Attrib redCSPerMin    -2.18372627276877
Sigmoid Node 4
    Inputs    Weights
    Threshold    -7.210722271541715
    Attrib blueFirstBlood=FALSE    -1.8797996883730936
    Attrib blueEliteMonsters    1.5895330251915807
    Attrib blueDragons=TRUE    -0.1280056160651928
    Attrib blueTotalGold    -2.1485344900369685
    Attrib blueTotalExperience    4.439898829022466
    Attrib blueGoldDiff    4.861782254397936
    Attrib blueCSPerMin    -3.3040107743064118
    Attrib redFirstBlood=TRUE    -1.947967266360278
    Attrib redEliteMonsters    3.7202980314764154
    Attrib redDragons=TRUE    -2.7974730262134604
    Attrib redTotalGold    -11.810296797242378
    Attrib redTotalExperience    -7.357000786884058
    Attrib redGoldDiff    -4.915798230855615
    Attrib redCSPerMin    -11.15621405228082
Sigmoid Node 5
    Inputs    Weights
    Threshold    -11.487726304600965
    Attrib blueFirstBlood=FALSE    0.3504067835031053
    Attrib blueEliteMonsters    4.820131593291832
    Attrib blueDragons=TRUE    1.1036326552234959
    Attrib blueTotalGold    -6.134375709972104
    Attrib blueTotalExperience    13.41535233413009
    Attrib blueGoldDiff    2.1206302774283183
    Attrib blueCSPerMin    -4.040364021537203
    Attrib redFirstBlood=TRUE    0.35310191688797554
    Attrib redEliteMonsters    -0.3587185587941613
    Attrib redDragons=TRUE    -1.8662375891832124
    Attrib redTotalGold    -10.997215276569579
    Attrib redTotalExperience    -2.6139560394211125
    Attrib redGoldDiff    -2.133108186003303
    Attrib redCSPerMin    -0.4944192116621434
Sigmoid Node 6
    Inputs    Weights
    Threshold    -11.810497927513751
    Attrib blueFirstBlood=FALSE    0.94297719139047
    Attrib blueEliteMonsters    -1.1162518909498826
    Attrib blueDragons=TRUE    1.1419414799321637
    Attrib blueTotalGold    9.998390850277824
    Attrib blueTotalExperience    3.679696421256237
    Attrib blueGoldDiff    4.928371291913049
    Attrib blueCSPerMin    7.975347675511298
    Attrib redFirstBlood=TRUE    0.9564062973829277
    Attrib redEliteMonsters    -3.50093083108605
    Attrib redDragons=TRUE    -0.15720538983907265
    Attrib redTotalGold    1.8064508776410007
    Attrib redTotalExperience    -8.860687685893238
    Attrib redGoldDiff    -4.976604368536057
    Attrib redCSPerMin    3.9343581131043672
Sigmoid Node 7
    Inputs    Weights
    Threshold    -15.950476443956404
    Attrib blueFirstBlood=FALSE    -0.7264301206319903
    Attrib blueEliteMonsters    2.350270599685101
    Attrib blueDragons=TRUE    -4.954122067875707
    Attrib blueTotalGold    -10.84227139602112
    Attrib blueTotalExperience    -1.4108316958335747
    Attrib blueGoldDiff    -6.563897312503132
    Attrib blueCSPerMin    10.079765771908287
    Attrib redFirstBlood=TRUE    -0.8230934490724684
    Attrib redEliteMonsters    4.063913181510177
    Attrib redDragons=TRUE    -0.7825991943400182
    Attrib redTotalGold    0.7187202400257062
    Attrib redTotalExperience    16.19439603929679
    Attrib redGoldDiff    6.525591466499703
    Attrib redCSPerMin    -1.81144960504719
Sigmoid Node 8
    Inputs    Weights
    Threshold    -5.3704977354233066
    Attrib blueFirstBlood=FALSE    2.617427602184948
    Attrib blueEliteMonsters    0.29003798334032377
    Attrib blueDragons=TRUE    0.26037761553451805
    Attrib blueTotalGold    -9.98639870388995
    Attrib blueTotalExperience    -16.61181458489764
    Attrib blueGoldDiff    -10.032163435705284
    Attrib blueCSPerMin    3.8225253432139645
    Attrib redFirstBlood=TRUE    2.5400878259086332
    Attrib redEliteMonsters    0.02368709344043355
    Attrib redDragons=TRUE    1.2236134384347357
    Attrib redTotalGold    8.339055302044413
    Attrib redTotalExperience    10.28487891625706
    Attrib redGoldDiff    10.084137746657161
    Attrib redCSPerMin    -7.688769076558862
Sigmoid Node 9
    Inputs    Weights
    Threshold    -8.140116378803336
    Attrib blueFirstBlood=FALSE    -0.4318252814960838
    Attrib blueEliteMonsters    3.5589852712337957
    Attrib blueDragons=TRUE    -0.45079191798958573
    Attrib blueTotalGold    -6.7104608868201545
    Attrib blueTotalExperience    -5.6483349669995055
    Attrib blueGoldDiff    -7.603459269265206
    Attrib blueCSPerMin    5.990930557562433
    Attrib redFirstBlood=TRUE    -0.47342562046608677
    Attrib redEliteMonsters    1.4961378985385891
    Attrib redDragons=TRUE    -1.1132344219700756
    Attrib redTotalGold    7.294718221040438
    Attrib redTotalExperience    -9.630016359989055
    Attrib redGoldDiff    7.591029854000683
    Attrib redCSPerMin    2.6424275206517596
Class FALSE
    Input
    Node 0
Class TRUE
    Input
    Node 1


Time taken to build model: 9.42 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        7101               71.8797 %
Incorrectly Classified Instances      2778               28.1203 %
Kappa statistic                          0.4377
Mean absolute error                      0.3513
Root mean squared error                  0.4286
Relative absolute error                 70.2542 %
Root relative squared error             85.7103 %
Total Number of Instances             9879     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,693    0,255    0,732      0,693    0,712      0,438    0,800     0,796     FALSE
                 0,745    0,307    0,707      0,745    0,726      0,438    0,800     0,797     TRUE
Weighted Avg.    0,719    0,281    0,719      0,719    0,719      0,438    0,800     0,796     

=== Confusion Matrix ===

    a    b   <-- classified as
 3428 1521 |    a = FALSE
 1257 3673 |    b = TRUE
```

#### Random Forest

```shell
=== Classifier model (full training set) ===

RandomForest

Bagging with 100 iterations and base learner

weka.classifiers.trees.RandomTree -K 0 -M 1.0 -V 0.001 -S 1 -do-not-check-capabilities

Time taken to build model: 2.93 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        7103               71.9    %
Incorrectly Classified Instances      2776               28.1    %
Kappa statistic                          0.438 
Mean absolute error                      0.3576
Root mean squared error                  0.4296
Relative absolute error                 71.5113 %
Root relative squared error             85.9124 %
Total Number of Instances             9879     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,730    0,292    0,715      0,730    0,722      0,438    0,796     0,791     FALSE
                 0,708    0,270    0,723      0,708    0,716      0,438    0,796     0,791     TRUE
Weighted Avg.    0,719    0,281    0,719      0,719    0,719      0,438    0,796     0,791     

=== Confusion Matrix ===

    a    b   <-- classified as
 3611 1338 |    a = FALSE
 1438 3492 |    b = TRUE
```
#### J48 (ID30)

```shell
=== Classifier model (full training set) ===

J48 pruned tree
------------------

blueGoldDiff <= 211
|   blueGoldDiff <= -1930: FALSE (2053.0/282.0)
|   blueGoldDiff > -1930
|   |   blueDragons = FALSE
|   |   |   blueGoldDiff <= -858
|   |   |   |   blueEliteMonsters <= 0
|   |   |   |   |   redCSPerMin <= 25.1: FALSE (848.0/210.0)
|   |   |   |   |   redCSPerMin > 25.1
|   |   |   |   |   |   redCSPerMin <= 26.5
|   |   |   |   |   |   |   blueTotalGold <= 15583: FALSE (38.0/9.0)
|   |   |   |   |   |   |   blueTotalGold > 15583
|   |   |   |   |   |   |   |   redCSPerMin <= 25.8: TRUE (13.0/4.0)
|   |   |   |   |   |   |   |   redCSPerMin > 25.8: FALSE (6.0/1.0)
|   |   |   |   |   |   redCSPerMin > 26.5: TRUE (6.0/2.0)
|   |   |   |   blueEliteMonsters > 0
|   |   |   |   |   redEliteMonsters <= 0
|   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   redTotalExperience <= 17986: FALSE (4.0)
|   |   |   |   |   |   |   redTotalExperience > 17986
|   |   |   |   |   |   |   |   blueCSPerMin <= 18.8: FALSE (2.0)
|   |   |   |   |   |   |   |   blueCSPerMin > 18.8
|   |   |   |   |   |   |   |   |   blueCSPerMin <= 22.6: TRUE (6.0)
|   |   |   |   |   |   |   |   |   blueCSPerMin > 22.6: FALSE (2.0)
|   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   blueTotalGold <= 15558
|   |   |   |   |   |   |   |   blueCSPerMin <= 19: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   blueCSPerMin > 19: FALSE (9.0)
|   |   |   |   |   |   |   blueTotalGold > 15558: TRUE (10.0/2.0)
|   |   |   |   |   redEliteMonsters > 0
|   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   blueCSPerMin <= 23.5
|   |   |   |   |   |   |   |   blueTotalGold <= 17478: FALSE (42.0/10.0)
|   |   |   |   |   |   |   |   blueTotalGold > 17478: TRUE (3.0)
|   |   |   |   |   |   |   blueCSPerMin > 23.5: TRUE (3.0)
|   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   redTotalGold <= 17904
|   |   |   |   |   |   |   |   blueTotalGold <= 16035: FALSE (46.0/9.0)
|   |   |   |   |   |   |   |   blueTotalGold > 16035
|   |   |   |   |   |   |   |   |   redCSPerMin <= 22.1: TRUE (8.0)
|   |   |   |   |   |   |   |   |   redCSPerMin > 22.1
|   |   |   |   |   |   |   |   |   |   redCSPerMin <= 22.8: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   redCSPerMin > 22.8: TRUE (4.0/1.0)
|   |   |   |   |   |   |   redTotalGold > 17904: FALSE (15.0)
|   |   |   blueGoldDiff > -858
|   |   |   |   redEliteMonsters <= 0: FALSE (315.0/150.0)
|   |   |   |   redEliteMonsters > 0
|   |   |   |   |   redTotalExperience <= 18957
|   |   |   |   |   |   redEliteMonsters <= 1
|   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   redTotalGold <= 18189
|   |   |   |   |   |   |   |   |   blueTotalExperience <= 18551
|   |   |   |   |   |   |   |   |   |   redCSPerMin <= 19: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   redCSPerMin > 19
|   |   |   |   |   |   |   |   |   |   |   redCSPerMin <= 24.6
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalExperience <= 17080
|   |   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin <= 21.9
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin <= 21.2: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin > 21.2: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin > 21.9: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalExperience > 17080: FALSE (19.0)
|   |   |   |   |   |   |   |   |   |   |   redCSPerMin > 24.6: TRUE (2.0)
|   |   |   |   |   |   |   |   |   blueTotalExperience > 18551: TRUE (3.0)
|   |   |   |   |   |   |   |   redTotalGold > 18189: TRUE (5.0)
|   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   blueEliteMonsters <= 0: FALSE (385.0/155.0)
|   |   |   |   |   |   |   |   blueEliteMonsters > 0
|   |   |   |   |   |   |   |   |   blueTotalGold <= 15092: FALSE (11.0)
|   |   |   |   |   |   |   |   |   blueTotalGold > 15092
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueTotalExperience <= 18766: TRUE (40.0/19.0)
|   |   |   |   |   |   |   |   |   |   |   blueTotalExperience > 18766: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -473
|   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin <= 19.1: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin > 19.1: FALSE (16.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -473: TRUE (42.0/17.0)
|   |   |   |   |   |   redEliteMonsters > 1
|   |   |   |   |   |   |   blueTotalGold <= 15912
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   redCSPerMin <= 22.7
|   |   |   |   |   |   |   |   |   |   blueTotalGold <= 15767: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueTotalGold > 15767: FALSE (2.0)
|   |   |   |   |   |   |   |   |   redCSPerMin > 22.7: FALSE (8.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   blueCSPerMin <= 21.5: FALSE (9.0)
|   |   |   |   |   |   |   |   |   blueCSPerMin > 21.5
|   |   |   |   |   |   |   |   |   |   redTotalGold <= 16001
|   |   |   |   |   |   |   |   |   |   |   blueTotalExperience <= 17148: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   blueTotalExperience > 17148
|   |   |   |   |   |   |   |   |   |   |   |   blueCSPerMin <= 22.9: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueCSPerMin > 22.9
|   |   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin <= 23.5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalExperience <= 17926: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalExperience > 17926: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin > 23.5: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   redTotalGold > 16001: FALSE (6.0)
|   |   |   |   |   |   |   blueTotalGold > 15912
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   redTotalExperience <= 18003: TRUE (12.0/1.0)
|   |   |   |   |   |   |   |   |   redTotalExperience > 18003: FALSE (13.0/3.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   blueCSPerMin <= 22.2
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -169: TRUE (10.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > -169
|   |   |   |   |   |   |   |   |   |   |   blueTotalGold <= 16431: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   blueTotalGold > 16431: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   blueCSPerMin > 22.2
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -333: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > -333: FALSE (11.0/2.0)
|   |   |   |   |   redTotalExperience > 18957
|   |   |   |   |   |   blueEliteMonsters <= 0
|   |   |   |   |   |   |   blueCSPerMin <= 24.2
|   |   |   |   |   |   |   |   redTotalGold <= 16975
|   |   |   |   |   |   |   |   |   redCSPerMin <= 21.7: FALSE (9.0)
|   |   |   |   |   |   |   |   |   redCSPerMin > 21.7
|   |   |   |   |   |   |   |   |   |   redEliteMonsters <= 1
|   |   |   |   |   |   |   |   |   |   |   redTotalGold <= 16742: FALSE (23.0/5.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalGold > 16742: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   redEliteMonsters > 1
|   |   |   |   |   |   |   |   |   |   |   blueCSPerMin <= 22.6: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   blueCSPerMin > 22.6: TRUE (5.0)
|   |   |   |   |   |   |   |   redTotalGold > 16975: FALSE (27.0/1.0)
|   |   |   |   |   |   |   blueCSPerMin > 24.2: FALSE (21.0)
|   |   |   |   |   |   blueEliteMonsters > 0
|   |   |   |   |   |   |   blueCSPerMin <= 22.1
|   |   |   |   |   |   |   |   blueTotalGold <= 16703: TRUE (7.0)
|   |   |   |   |   |   |   |   blueTotalGold > 16703
|   |   |   |   |   |   |   |   |   redTotalExperience <= 19312: TRUE (3.0)
|   |   |   |   |   |   |   |   |   redTotalExperience > 19312: FALSE (4.0)
|   |   |   |   |   |   |   blueCSPerMin > 22.1: FALSE (12.0/2.0)
|   |   blueDragons = TRUE
|   |   |   blueCSPerMin <= 17.4: FALSE (28.0/4.0)
|   |   |   blueCSPerMin > 17.4
|   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   blueEliteMonsters <= 1
|   |   |   |   |   |   redEliteMonsters <= 0
|   |   |   |   |   |   |   blueGoldDiff <= 85: FALSE (279.0/130.0)
|   |   |   |   |   |   |   blueGoldDiff > 85
|   |   |   |   |   |   |   |   blueGoldDiff <= 176: TRUE (24.0/2.0)
|   |   |   |   |   |   |   |   blueGoldDiff > 176
|   |   |   |   |   |   |   |   |   blueTotalGold <= 16246
|   |   |   |   |   |   |   |   |   |   blueTotalGold <= 15786: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueTotalGold > 15786: TRUE (2.0)
|   |   |   |   |   |   |   |   |   blueTotalGold > 16246: FALSE (3.0)
|   |   |   |   |   |   redEliteMonsters > 0
|   |   |   |   |   |   |   redCSPerMin <= 23.7: FALSE (81.0/37.0)
|   |   |   |   |   |   |   redCSPerMin > 23.7: TRUE (16.0/2.0)
|   |   |   |   |   blueEliteMonsters > 1: TRUE (86.0/35.0)
|   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   redTotalGold <= 15998
|   |   |   |   |   |   blueTotalGold <= 14609: FALSE (36.0/12.0)
|   |   |   |   |   |   blueTotalGold > 14609
|   |   |   |   |   |   |   redTotalExperience <= 18720: TRUE (73.0/17.0)
|   |   |   |   |   |   |   redTotalExperience > 18720
|   |   |   |   |   |   |   |   redTotalExperience <= 19065: FALSE (11.0/2.0)
|   |   |   |   |   |   |   |   redTotalExperience > 19065: TRUE (7.0/1.0)
|   |   |   |   |   redTotalGold > 15998
|   |   |   |   |   |   redEliteMonsters <= 0
|   |   |   |   |   |   |   blueEliteMonsters <= 1: FALSE (292.0/122.0)
|   |   |   |   |   |   |   blueEliteMonsters > 1
|   |   |   |   |   |   |   |   blueTotalExperience <= 19071: FALSE (49.0/17.0)
|   |   |   |   |   |   |   |   blueTotalExperience > 19071: TRUE (5.0)
|   |   |   |   |   |   redEliteMonsters > 0
|   |   |   |   |   |   |   blueCSPerMin <= 21.9: FALSE (37.0/8.0)
|   |   |   |   |   |   |   blueCSPerMin > 21.9
|   |   |   |   |   |   |   |   redTotalGold <= 17155
|   |   |   |   |   |   |   |   |   blueGoldDiff <= -951
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -1399: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > -1399: TRUE (5.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > -951
|   |   |   |   |   |   |   |   |   |   redCSPerMin <= 21.6
|   |   |   |   |   |   |   |   |   |   |   blueTotalGold <= 16435: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   blueTotalGold > 16435: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   redCSPerMin > 21.6: FALSE (10.0)
|   |   |   |   |   |   |   |   redTotalGold > 17155: TRUE (10.0/1.0)
blueGoldDiff > 211
|   blueGoldDiff <= 1769
|   |   redDragons = FALSE: TRUE (1470.0/485.0)
|   |   redDragons = TRUE
|   |   |   blueGoldDiff <= 828
|   |   |   |   redEliteMonsters <= 1
|   |   |   |   |   blueCSPerMin <= 19.2
|   |   |   |   |   |   redTotalGold <= 16557: FALSE (16.0)
|   |   |   |   |   |   redTotalGold > 16557
|   |   |   |   |   |   |   blueCSPerMin <= 18.8: TRUE (4.0)
|   |   |   |   |   |   |   blueCSPerMin > 18.8: FALSE (4.0)
|   |   |   |   |   blueCSPerMin > 19.2
|   |   |   |   |   |   blueEliteMonsters <= 0
|   |   |   |   |   |   |   blueTotalExperience <= 18838
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE: FALSE (95.0/41.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   blueCSPerMin <= 22.2
|   |   |   |   |   |   |   |   |   |   blueTotalExperience <= 17761
|   |   |   |   |   |   |   |   |   |   |   redTotalExperience <= 17789
|   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin <= 17.5: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   redCSPerMin > 17.5: TRUE (12.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalExperience > 17789: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   blueTotalExperience > 17761: FALSE (21.0/2.0)
|   |   |   |   |   |   |   |   |   blueCSPerMin > 22.2: TRUE (42.0/14.0)
|   |   |   |   |   |   |   blueTotalExperience > 18838
|   |   |   |   |   |   |   |   blueCSPerMin <= 23.3: TRUE (31.0/4.0)
|   |   |   |   |   |   |   |   blueCSPerMin > 23.3
|   |   |   |   |   |   |   |   |   redTotalGold <= 16031: FALSE (5.0)
|   |   |   |   |   |   |   |   |   redTotalGold > 16031: TRUE (13.0/4.0)
|   |   |   |   |   |   blueEliteMonsters > 0
|   |   |   |   |   |   |   blueFirstBlood = TRUE: FALSE (58.0/26.0)
|   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   blueTotalGold <= 16592
|   |   |   |   |   |   |   |   |   blueTotalExperience <= 17975
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 462: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > 462: FALSE (2.0)
|   |   |   |   |   |   |   |   |   blueTotalExperience > 17975: FALSE (11.0)
|   |   |   |   |   |   |   |   blueTotalGold > 16592
|   |   |   |   |   |   |   |   |   blueGoldDiff <= 439
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 291: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > 291: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > 439: TRUE (8.0)
|   |   |   |   redEliteMonsters > 1
|   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   blueCSPerMin <= 24.7
|   |   |   |   |   |   |   redCSPerMin <= 20.7
|   |   |   |   |   |   |   |   redCSPerMin <= 19.9: FALSE (3.0)
|   |   |   |   |   |   |   |   redCSPerMin > 19.9
|   |   |   |   |   |   |   |   |   blueCSPerMin <= 23.4: TRUE (3.0)
|   |   |   |   |   |   |   |   |   blueCSPerMin > 23.4: FALSE (2.0)
|   |   |   |   |   |   |   redCSPerMin > 20.7: FALSE (11.0)
|   |   |   |   |   |   blueCSPerMin > 24.7: TRUE (3.0)
|   |   |   |   |   blueFirstBlood = FALSE: FALSE (24.0/11.0)
|   |   |   blueGoldDiff > 828
|   |   |   |   blueEliteMonsters <= 0: TRUE (343.0/123.0)
|   |   |   |   blueEliteMonsters > 0
|   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   redTotalGold <= 15989: TRUE (54.0/18.0)
|   |   |   |   |   |   redTotalGold > 15989
|   |   |   |   |   |   |   redTotalGold <= 17099: FALSE (33.0/9.0)
|   |   |   |   |   |   |   redTotalGold > 17099: TRUE (4.0)
|   |   |   |   |   blueFirstBlood = FALSE: TRUE (28.0/11.0)
|   blueGoldDiff > 1769: TRUE (2283.0/325.0)

Number of Leaves  :     120

Size of the tree :  239


Time taken to build model: 0.31 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        7040               71.2623 %
Incorrectly Classified Instances      2839               28.7377 %
Kappa statistic                          0.4252
Mean absolute error                      0.3646
Root mean squared error                  0.448 
Relative absolute error                 72.9142 %
Root relative squared error             89.6098 %
Total Number of Instances             9879     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,725    0,300    0,708      0,725    0,717      0,425    0,762     0,723     FALSE
                 0,700    0,275    0,717      0,700    0,709      0,425    0,762     0,713     TRUE
Weighted Avg.    0,713    0,287    0,713      0,713    0,713      0,425    0,762     0,718     

=== Confusion Matrix ===

    a    b   <-- classified as
 3588 1361 |    a = FALSE
 1478 3452 |    b = TRUE
```

#### Decision Stump

``` shell
=== Classifier model (full training set) ===

Decision Stump

Classifications

blueGoldDiff <= 212.0 : FALSE
blueGoldDiff > 212.0 : TRUE
blueGoldDiff is missing : FALSE

Class distributions

blueGoldDiff <= 212.0
FALSE   TRUE    
0.7116878196628149  0.2883121803371851  
blueGoldDiff > 212.0
FALSE   TRUE    
0.2591304347826087  0.7408695652173913  
blueGoldDiff is missing
FALSE   TRUE    
0.5009616357930965  0.4990383642069035  


Time taken to build model: 0.04 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        7154               72.4162 %
Incorrectly Classified Instances      2725               27.5838 %
Kappa statistic                          0.4483
Mean absolute error                      0.3986
Root mean squared error                  0.4468
Relative absolute error                 79.7221 %
Root relative squared error             89.3594 %
Total Number of Instances             9879     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,757    0,308    0,711      0,757    0,733      0,449    0,717     0,670     FALSE
                 0,692    0,243    0,739      0,692    0,715      0,449    0,717     0,682     TRUE
Weighted Avg.    0,724    0,276    0,725      0,724    0,724      0,449    0,717     0,676     

=== Confusion Matrix ===

    a    b   <-- classified as
 3744 1205 |    a = FALSE
 1520 3410 |    b = TRUE
```