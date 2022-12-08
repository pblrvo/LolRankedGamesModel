# APRENDIZAJE AUTOMÁTICO LolRankedGamesModel

## SOLUCIONES DISTINTOS ALGORITMOS
### RANDOM FOREST

```shell
=== Run information ===

Scheme:       weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1
Relation:     datos_simplificados
Instances:    9879
Attributes:   23
              blueWins
              blueFirstBlood
              blueKills
              blueDeaths
              blueAssists
              blueDragons
              blueHeralds
              blueTowersDestroyed
              blueAvgLevel
              blueTotalMinionsKilled
              blueGoldDiff
              blueExperienceDiff
              redFirstBlood
              redKills
              redDeaths
              redAssists
              redDragons
              redHeralds
              redTowersDestroyed
              redAvgLevel
              redTotalMinionsKilled
              redGoldDiff
              redExperienceDiff
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

RandomForest

Bagging with 100 iterations and base learner

weka.classifiers.trees.RandomTree -K 0 -M 1.0 -V 0.001 -S 1 -do-not-check-capabilities

Time taken to build model: 3.05 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        7098               71.8494 %
Incorrectly Classified Instances      2781               28.1506 %
Kappa statistic                          0.437 
Mean absolute error                      0.3605
Root mean squared error                  0.43  
Relative absolute error                 72.1088 %
Root relative squared error             85.997  %
Total Number of Instances             9879     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,732    0,295    0,714      0,732    0,723      0,437    0,795     0,790     FALSE
                 0,705    0,268    0,724      0,705    0,714      0,437    0,795     0,788     TRUE
Weighted Avg.    0,718    0,282    0,719      0,718    0,718      0,437    0,795     0,789     

=== Confusion Matrix ===

    a    b   <-- classified as
 3622 1327 |    a = FALSE
 1454 3476 |    b = TRUE
```
### J48(ID3)
```shell
=== Run information ===

Scheme:       weka.classifiers.trees.J48 -C 0.25 -M 2
Relation:     datos_simplificados
Instances:    9879
Attributes:   23
              blueWins
              blueFirstBlood
              blueKills
              blueDeaths
              blueAssists
              blueDragons
              blueHeralds
              blueTowersDestroyed
              blueAvgLevel
              blueTotalMinionsKilled
              blueGoldDiff
              blueExperienceDiff
              redFirstBlood
              redKills
              redDeaths
              redAssists
              redDragons
              redHeralds
              redTowersDestroyed
              redAvgLevel
              redTotalMinionsKilled
              redGoldDiff
              redExperienceDiff
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

J48 pruned tree
------------------

blueGoldDiff <= 211
|   blueGoldDiff <= -1930: FALSE (2053.0/282.0)
|   blueGoldDiff > -1930
|   |   blueDragons = FALSE
|   |   |   blueGoldDiff <= -858
|   |   |   |   redTowersDestroyed <= 0
|   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   redAvgLevel <= 6.6
|   |   |   |   |   |   |   redAssists <= 13
|   |   |   |   |   |   |   |   blueDeaths <= 4: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   blueDeaths > 4
|   |   |   |   |   |   |   |   |   blueGoldDiff <= -1679: TRUE (4.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > -1679
|   |   |   |   |   |   |   |   |   |   redAssists <= 11: FALSE (31.0/8.0)
|   |   |   |   |   |   |   |   |   |   redAssists > 11: TRUE (6.0/2.0)
|   |   |   |   |   |   |   redAssists > 13: FALSE (9.0)
|   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   blueAvgLevel <= 6.6
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -2280: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -2280: TRUE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.4: FALSE (13.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.4
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 8
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 3: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 3: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 4: FALSE (25.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 8
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 205: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 205: TRUE (3.0)
|   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   blueKills <= 4
|   |   |   |   |   |   |   |   |   |   |   redAssists <= 3: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   redAssists > 3: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 4: FALSE (7.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE: FALSE (123.0/20.0)
|   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   blueAssists <= 6: FALSE (98.0/17.0)
|   |   |   |   |   |   |   |   |   |   blueAssists > 6
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 10: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 10: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -1672: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -1672: FALSE (8.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 6: FALSE (9.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueKills > 6
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 8: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 8: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 8: TRUE (3.0)
|   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 0: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 0
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 12: FALSE (35.0/6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 12: TRUE (6.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -1027: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -1027
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 9
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 257: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 257: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 4: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 6: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 9: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: FALSE (11.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7.2
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 2: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 2: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 7: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 192
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 221: TRUE (9.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 221
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 190: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 190: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 192
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 249: FALSE (197.0/40.0)
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 249
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 210: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 210
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -1436: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -1436: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 6: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -1640
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 226: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 226: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -1640: TRUE (10.0)
|   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7: FALSE (103.0/18.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -1623: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -1623: FALSE (7.0)
|   |   |   |   |   |   |   |   |   redAvgLevel > 7.2
|   |   |   |   |   |   |   |   |   |   redDragons = FALSE: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueKills <= 6: FALSE (11.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueKills > 6: TRUE (2.0)
|   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   blueAssists <= 1: FALSE (11.0)
|   |   |   |   |   |   blueAssists > 1
|   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   blueKills <= 6
|   |   |   |   |   |   |   |   |   blueAssists <= 4
|   |   |   |   |   |   |   |   |   |   blueAssists <= 3
|   |   |   |   |   |   |   |   |   |   |   redAssists <= 2: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   redAssists > 2: FALSE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueAssists > 3
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 208: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 208: TRUE (6.0)
|   |   |   |   |   |   |   |   |   blueAssists > 4: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   blueKills > 6: TRUE (8.0/1.0)
|   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   blueTowersDestroyed <= 0
|   |   |   |   |   |   |   |   |   blueExperienceDiff <= -474: FALSE (72.0/16.0)
|   |   |   |   |   |   |   |   |   blueExperienceDiff > -474
|   |   |   |   |   |   |   |   |   |   blueKills <= 7
|   |   |   |   |   |   |   |   |   |   |   blueAssists <= 6
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 2: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 2: FALSE (14.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   blueAssists > 6: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 7: FALSE (6.0/2.0)
|   |   |   |   |   |   |   |   blueTowersDestroyed > 0
|   |   |   |   |   |   |   |   |   blueAssists <= 5: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   blueAssists > 5: FALSE (2.0)
|   |   |   |   redTowersDestroyed > 0
|   |   |   |   |   redTowersDestroyed <= 1
|   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   redDragons = FALSE: FALSE (5.0/1.0)
|   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   blueKills <= 6: TRUE (5.0)
|   |   |   |   |   |   |   |   blueKills > 6
|   |   |   |   |   |   |   |   |   blueHeralds = FALSE: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   blueHeralds = TRUE: FALSE (2.0)
|   |   |   |   |   |   redHeralds > 0: FALSE (16.0/2.0)
|   |   |   |   |   redTowersDestroyed > 1: TRUE (2.0)
|   |   |   blueGoldDiff > -858
|   |   |   |   blueDeaths <= 3
|   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   redAssists <= 1: TRUE (8.0/2.0)
|   |   |   |   |   |   |   redAssists > 1
|   |   |   |   |   |   |   |   blueDeaths <= 2: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   blueDeaths > 2
|   |   |   |   |   |   |   |   |   blueGoldDiff <= -393: TRUE (3.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > -393
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 103
|   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 396: FALSE (7.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 396: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > 103: TRUE (2.0)
|   |   |   |   |   |   redAvgLevel > 7: FALSE (4.0)
|   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   blueAvgLevel <= 6.6
|   |   |   |   |   |   |   blueExperienceDiff <= -494: TRUE (5.0)
|   |   |   |   |   |   |   blueExperienceDiff > -494: FALSE (3.0)
|   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   redAssists <= 2
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE: FALSE (16.0/4.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   blueAssists <= 2: FALSE (7.0/1.0)
|   |   |   |   |   |   |   |   |   blueAssists > 2: TRUE (6.0/1.0)
|   |   |   |   |   |   |   redAssists > 2: FALSE (37.0/2.0)
|   |   |   |   blueDeaths > 3
|   |   |   |   |   blueExperienceDiff <= -906
|   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   blueAssists <= 12
|   |   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 10
|   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 3: TRUE (6.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAssists > 3
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -659: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -659: FALSE (14.0/3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.6: FALSE (17.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 10: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   redHeralds > 0: FALSE (8.0)
|   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.6
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -1616: FALSE (10.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -1616
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 219: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 219: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE: FALSE (13.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 6: FALSE (18.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 226
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 3: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 3
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -1911: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -1911: FALSE (15.0/3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 7: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 7: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 226: FALSE (9.0)
|   |   |   |   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 3: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 6: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 6: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 7.2: FALSE (6.0)
|   |   |   |   |   |   |   |   blueAssists > 12
|   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.4: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.4: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   redHeralds > 0: TRUE (2.0)
|   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 204: TRUE (5.0)
|   |   |   |   |   |   |   |   blueTotalMinionsKilled > 204
|   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 5: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 5: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7.2: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   redAssists <= 4: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   redAssists > 4
|   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 8: TRUE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAssists > 8: FALSE (2.0)
|   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 227: FALSE (7.0)
|   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 227: TRUE (4.0/1.0)
|   |   |   |   |   |   blueAvgLevel > 7: FALSE (12.0)
|   |   |   |   |   blueExperienceDiff > -906
|   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.6
|   |   |   |   |   |   |   |   |   |   blueKills <= 4: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 4
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 6: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 6: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.4: FALSE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueKills > 8: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8: TRUE (3.0)
|   |   |   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   blueKills <= 3: FALSE (7.0/2.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   |   blueKills <= 4: TRUE (13.0)
|   |   |   |   |   |   |   |   |   |   |   blueKills > 4
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 218: TRUE (9.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 218: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 10
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 753: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 753: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 5: TRUE (8.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 9: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 9: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8: FALSE (13.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 6: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 6: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 6: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 7: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 10: TRUE (3.0)
|   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   redAssists <= 5: FALSE (46.0/15.0)
|   |   |   |   |   |   |   |   |   redAssists > 5
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 515: TRUE (52.0/17.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 515: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 205
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 184: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 184: FALSE (9.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 205: TRUE (8.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 8: FALSE (13.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 5: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 212: FALSE (7.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 212: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.6: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 8: TRUE (11.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 8: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 235: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 235: FALSE (3.0/1.0)
|   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: TRUE (5.0/2.0)
|   |   |   |   |   |   |   |   |   redAvgLevel > 6.6: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   blueKills <= 6
|   |   |   |   |   |   |   |   |   |   redAssists <= 3: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   redAssists > 3
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 204: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 204: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6: FALSE (22.0/5.0)
|   |   |   |   |   |   |   |   |   blueKills > 6: TRUE (7.0/1.0)
|   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   redTotalMinionsKilled <= 192
|   |   |   |   |   |   |   |   blueAssists <= 10: FALSE (21.0/1.0)
|   |   |   |   |   |   |   |   blueAssists > 10: TRUE (2.0)
|   |   |   |   |   |   |   redTotalMinionsKilled > 192
|   |   |   |   |   |   |   |   blueAssists <= 9
|   |   |   |   |   |   |   |   |   blueAssists <= 8
|   |   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 758
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 233
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 217: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 217
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 170: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 170: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 233: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 758: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 11
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 219
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 5: FALSE (9.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 209: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 209: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 219
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 221
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 7: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 7: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 221: TRUE (7.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: TRUE (8.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 6: TRUE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 11: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 3
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 498: FALSE (22.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 498: TRUE (8.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -653: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -653: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 5: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 4: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 4: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 1: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 1
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 216: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 216: TRUE (14.0/3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7: FALSE (53.0/17.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -357: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -357: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE: TRUE (17.0/6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7.2: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 5: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 678: FALSE (12.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 678: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 8: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 206: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 206: FALSE (7.0)
|   |   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -482: FALSE (10.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -482
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 233
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE: TRUE (21.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7.2: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 233
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 218: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 218: FALSE (11.0/1.0)
|   |   |   |   |   |   |   |   |   blueAssists > 8
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7: TRUE (10.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   redDragons = FALSE: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redDragons = TRUE: FALSE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE: TRUE (2.0)
|   |   |   |   |   |   |   |   blueAssists > 9
|   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 238: FALSE (15.0/3.0)
|   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 238: TRUE (3.0)
|   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   blueKills <= 9: FALSE (18.0/2.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 9: TRUE (3.0/1.0)
|   |   blueDragons = TRUE
|   |   |   blueTotalMinionsKilled <= 174: FALSE (28.0/4.0)
|   |   |   blueTotalMinionsKilled > 174
|   |   |   |   blueExperienceDiff <= -741
|   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   blueDeaths <= 9
|   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   blueDeaths <= 8
|   |   |   |   |   |   |   |   |   redHeralds <= 0: FALSE (123.0/51.0)
|   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.4: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.4
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 10
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 228: FALSE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 228: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -1784
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -1784: TRUE (11.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 10: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   blueDeaths > 8
|   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 217: FALSE (19.0/1.0)
|   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 217: TRUE (4.0/1.0)
|   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   redAssists <= 7
|   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.6: FALSE (4.0)
|   |   |   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   blueDeaths <= 5: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueDeaths > 5: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   redAssists > 7: TRUE (11.0/1.0)
|   |   |   |   |   |   blueDeaths > 9
|   |   |   |   |   |   |   blueKills <= 7: FALSE (4.0/1.0)
|   |   |   |   |   |   |   blueKills > 7
|   |   |   |   |   |   |   |   blueGoldDiff <= -605: TRUE (11.0)
|   |   |   |   |   |   |   |   blueGoldDiff > -605: FALSE (3.0/1.0)
|   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   blueAvgLevel <= 6.6: FALSE (44.0/19.0)
|   |   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   blueAssists <= 2: FALSE (18.0/1.0)
|   |   |   |   |   |   |   |   |   blueAssists > 2
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 9: FALSE (30.0/6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 9: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 198: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 198: FALSE (28.0/7.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: TRUE (4.0/1.0)
|   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   blueExperienceDiff <= -1353: FALSE (15.0/1.0)
|   |   |   |   |   |   |   |   blueExperienceDiff > -1353
|   |   |   |   |   |   |   |   |   blueKills <= 6
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   |   |   redAssists <= 9
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 3: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 3: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 5: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   redAssists > 9: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 7.2: TRUE (2.0)
|   |   |   |   |   |   |   |   |   blueKills > 6: FALSE (3.0)
|   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   redAssists <= 4: FALSE (3.0)
|   |   |   |   |   |   |   |   redAssists > 4: TRUE (7.0/1.0)
|   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 239: FALSE (8.0)
|   |   |   |   |   |   |   |   blueTotalMinionsKilled > 239: TRUE (2.0)
|   |   |   |   blueExperienceDiff > -741
|   |   |   |   |   redTowersDestroyed <= 0
|   |   |   |   |   |   blueDeaths <= 6
|   |   |   |   |   |   |   redAssists <= 9
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE: TRUE (132.0/47.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   redAssists <= 5
|   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 2: TRUE (13.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 2
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 2: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 2
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8: TRUE (19.0/4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 3: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 3: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -754: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -754: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -1097: FALSE (8.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -1097
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 1: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 1
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 219: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 219
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 514: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 514: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 2: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 2: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 4: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 7: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.6: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -337
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 2: FALSE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 2: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -337: TRUE (7.0)
|   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 228: TRUE (9.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 228: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   redAssists > 5
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: FALSE (9.0/1.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   blueKills <= 2: FALSE (9.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueKills > 2
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 226
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 200: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 200
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 6: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8: FALSE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 4: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 31: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 31: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redHeralds > 0: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 226: TRUE (25.0/5.0)
|   |   |   |   |   |   |   redAssists > 9: TRUE (24.0/4.0)
|   |   |   |   |   |   blueDeaths > 6
|   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   blueKills <= 4
|   |   |   |   |   |   |   |   |   blueKills <= 3
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE: FALSE (3.0)
|   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 208: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 208: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE: TRUE (10.0)
|   |   |   |   |   |   |   |   blueKills > 4
|   |   |   |   |   |   |   |   |   blueAssists <= 9
|   |   |   |   |   |   |   |   |   |   blueAssists <= 7
|   |   |   |   |   |   |   |   |   |   |   redAssists <= 12
|   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -485: TRUE (12.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -485
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 4: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 4
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 7: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 194: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 194: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE: FALSE (40.0/18.0)
|   |   |   |   |   |   |   |   |   |   |   redAssists > 12: FALSE (13.0/2.0)
|   |   |   |   |   |   |   |   |   |   blueAssists > 7: FALSE (44.0/10.0)
|   |   |   |   |   |   |   |   |   blueAssists > 9
|   |   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -428: TRUE (10.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -428
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 10: FALSE (10.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 10
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 232
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 9: TRUE (10.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 9: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 232: FALSE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   redHeralds > 0: TRUE (9.0/1.0)
|   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   blueKills <= 5: FALSE (8.0)
|   |   |   |   |   |   |   |   blueKills > 5
|   |   |   |   |   |   |   |   |   blueExperienceDiff <= -431: FALSE (6.0)
|   |   |   |   |   |   |   |   |   blueExperienceDiff > -431
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= -63: TRUE (8.0)
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > -63: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 224: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 224: TRUE (6.0/1.0)
|   |   |   |   |   redTowersDestroyed > 0
|   |   |   |   |   |   blueGoldDiff <= -1029: FALSE (9.0)
|   |   |   |   |   |   blueGoldDiff > -1029
|   |   |   |   |   |   |   redAssists <= 7
|   |   |   |   |   |   |   |   redHeralds <= 0: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 232: FALSE (7.0)
|   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 232: TRUE (2.0)
|   |   |   |   |   |   |   redAssists > 7: TRUE (5.0)
blueGoldDiff > 211
|   blueGoldDiff <= 1769
|   |   blueAvgLevel <= 6.6
|   |   |   blueTowersDestroyed <= 0
|   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   redDragons = FALSE
|   |   |   |   |   |   |   blueAvgLevel <= 6.4
|   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   blueDeaths <= 5: TRUE (4.0)
|   |   |   |   |   |   |   |   |   blueDeaths > 5
|   |   |   |   |   |   |   |   |   |   blueKills <= 8: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 8: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   redAvgLevel > 6.8: FALSE (3.0)
|   |   |   |   |   |   |   blueAvgLevel > 6.4
|   |   |   |   |   |   |   |   redAvgLevel <= 6.4: TRUE (11.0/2.0)
|   |   |   |   |   |   |   |   redAvgLevel > 6.4
|   |   |   |   |   |   |   |   |   blueKills <= 3: TRUE (4.0)
|   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   blueDragons = FALSE
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 212
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 8: FALSE (9.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 8: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 212: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   |   blueDragons = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 7: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 3: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 3: TRUE (7.0/1.0)
|   |   |   |   |   |   redDragons = TRUE
|   |   |   |   |   |   |   blueGoldDiff <= 614: FALSE (20.0/3.0)
|   |   |   |   |   |   |   blueGoldDiff > 614
|   |   |   |   |   |   |   |   blueAssists <= 3: TRUE (5.0)
|   |   |   |   |   |   |   |   blueAssists > 3
|   |   |   |   |   |   |   |   |   blueKills <= 4: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   blueKills > 4
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.4: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.4
|   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 173: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 173
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 234: TRUE (14.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 234: FALSE (2.0)
|   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   blueDragons = FALSE
|   |   |   |   |   |   |   blueKills <= 8: FALSE (3.0)
|   |   |   |   |   |   |   blueKills > 8
|   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 203: FALSE (2.0)
|   |   |   |   |   |   |   |   blueTotalMinionsKilled > 203: TRUE (2.0)
|   |   |   |   |   |   blueDragons = TRUE
|   |   |   |   |   |   |   blueTotalMinionsKilled <= 219: TRUE (3.0)
|   |   |   |   |   |   |   blueTotalMinionsKilled > 219: FALSE (3.0)
|   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   redAssists <= 10
|   |   |   |   |   |   blueAvgLevel <= 6.4: FALSE (10.0/2.0)
|   |   |   |   |   |   blueAvgLevel > 6.4
|   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   blueDeaths <= 7
|   |   |   |   |   |   |   |   |   redAssists <= 8
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.4: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.4: FALSE (21.0/5.0)
|   |   |   |   |   |   |   |   |   redAssists > 8: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   blueDeaths > 7: FALSE (4.0)
|   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   redDragons = FALSE: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   redDragons = TRUE: TRUE (2.0)
|   |   |   |   |   redAssists > 10: TRUE (4.0)
|   |   |   blueTowersDestroyed > 0
|   |   |   |   blueDeaths <= 6: TRUE (6.0)
|   |   |   |   blueDeaths > 6
|   |   |   |   |   blueFirstBlood = TRUE: FALSE (4.0)
|   |   |   |   |   blueFirstBlood = FALSE: TRUE (4.0/1.0)
|   |   blueAvgLevel > 6.6
|   |   |   redDragons = FALSE
|   |   |   |   redAvgLevel <= 6.4
|   |   |   |   |   blueDeaths <= 6: TRUE (51.0/4.0)
|   |   |   |   |   blueDeaths > 6
|   |   |   |   |   |   blueAssists <= 5: FALSE (3.0)
|   |   |   |   |   |   blueAssists > 5
|   |   |   |   |   |   |   blueAssists <= 12: TRUE (12.0/2.0)
|   |   |   |   |   |   |   blueAssists > 12: FALSE (2.0)
|   |   |   |   redAvgLevel > 6.4
|   |   |   |   |   blueAvgLevel <= 7.2
|   |   |   |   |   |   blueDragons = FALSE
|   |   |   |   |   |   |   blueExperienceDiff <= 885
|   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 203: FALSE (11.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 203
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 9
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 2
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 3: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 1101: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 1101: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 2
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 231
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 1273: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 1273: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 231: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 5: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 5: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 7: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 9: TRUE (9.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8: FALSE (55.0/23.0)
|   |   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE: TRUE (69.0/33.0)
|   |   |   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 234: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 234: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 201: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 201
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4: TRUE (7.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 201: TRUE (8.0)
|   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 201
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 7
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 616: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 616: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 7: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: TRUE (2.0)
|   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   blueDeaths <= 8
|   |   |   |   |   |   |   |   |   |   redHeralds <= 0: TRUE (30.0/4.0)
|   |   |   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 756: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 756: TRUE (2.0)
|   |   |   |   |   |   |   |   |   blueDeaths > 8: FALSE (3.0)
|   |   |   |   |   |   |   blueExperienceDiff > 885
|   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   blueDeaths <= 6: TRUE (81.0/12.0)
|   |   |   |   |   |   |   |   |   blueDeaths > 6
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   blueAssists <= 10
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 7: FALSE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 7: TRUE (7.0)
|   |   |   |   |   |   |   |   |   |   |   blueAssists > 10: TRUE (10.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 7: FALSE (2.0)
|   |   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   |   blueTowersDestroyed <= 0
|   |   |   |   |   |   |   |   |   |   blueDeaths <= 5
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 235: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 235: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 554: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 554: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   blueDeaths > 5: TRUE (8.0)
|   |   |   |   |   |   |   |   |   blueTowersDestroyed > 0: FALSE (3.0/1.0)
|   |   |   |   |   |   blueDragons = TRUE
|   |   |   |   |   |   |   blueAssists <= 4
|   |   |   |   |   |   |   |   blueAssists <= 1
|   |   |   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   |   |   redAssists <= 4: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   redAssists > 4: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   blueHeralds = TRUE: TRUE (3.0)
|   |   |   |   |   |   |   |   blueAssists > 1: TRUE (142.0/25.0)
|   |   |   |   |   |   |   blueAssists > 4
|   |   |   |   |   |   |   |   redHeralds <= 0: TRUE (516.0/172.0)
|   |   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   blueKills <= 9: TRUE (37.0/7.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 9
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 896: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 896: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 220: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 220
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= -391: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > -391: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 5: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueKills > 5: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 6: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 219: FALSE (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 219
|   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 364: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 364: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueKills > 5: TRUE (6.0/1.0)
|   |   |   |   |   blueAvgLevel > 7.2
|   |   |   |   |   |   blueAssists <= 5: TRUE (39.0/2.0)
|   |   |   |   |   |   blueAssists > 5
|   |   |   |   |   |   |   blueDragons = FALSE
|   |   |   |   |   |   |   |   redTotalMinionsKilled <= 226
|   |   |   |   |   |   |   |   |   blueGoldDiff <= 517: FALSE (6.0/1.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > 517: TRUE (14.0)
|   |   |   |   |   |   |   |   redTotalMinionsKilled > 226: FALSE (4.0)
|   |   |   |   |   |   |   blueDragons = TRUE
|   |   |   |   |   |   |   |   blueAssists <= 6: TRUE (12.0)
|   |   |   |   |   |   |   |   blueAssists > 6
|   |   |   |   |   |   |   |   |   blueDeaths <= 8
|   |   |   |   |   |   |   |   |   |   blueKills <= 9
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 2: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 2
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 304: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 304: TRUE (31.0/4.0)
|   |   |   |   |   |   |   |   |   |   blueKills > 9: FALSE (7.0/2.0)
|   |   |   |   |   |   |   |   |   blueDeaths > 8: TRUE (7.0)
|   |   |   redDragons = TRUE
|   |   |   |   redAvgLevel <= 7
|   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   blueGoldDiff <= 625
|   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   blueAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 341
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 5: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueKills > 5: FALSE (5.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueGoldDiff > 341: FALSE (19.0/2.0)
|   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 212: FALSE (13.0/2.0)
|   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 212
|   |   |   |   |   |   |   |   |   |   |   blueAssists <= 2: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   blueAssists > 2
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 2: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 2
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 9: TRUE (9.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 9
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 20: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 20: TRUE (2.0)
|   |   |   |   |   |   |   |   blueAvgLevel > 7.2: TRUE (4.0)
|   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   blueGoldDiff <= 573
|   |   |   |   |   |   |   |   |   redAssists <= 9
|   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 5
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 455
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 289: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 289: FALSE (8.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 455: TRUE (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 5: TRUE (11.0/1.0)
|   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 224
|   |   |   |   |   |   |   |   |   |   |   |   blueKills <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 236: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueTotalMinionsKilled > 236: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueKills > 6: FALSE (7.0)
|   |   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 224: TRUE (6.0)
|   |   |   |   |   |   |   |   |   redAssists > 9: TRUE (8.0/1.0)
|   |   |   |   |   |   |   |   blueGoldDiff > 573: FALSE (10.0/1.0)
|   |   |   |   |   |   blueGoldDiff > 625
|   |   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   redAssists <= 1: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   redAssists > 1
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 7
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: TRUE (22.0/4.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 7: FALSE (33.0/15.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 7
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 1288: TRUE (17.0/2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 1288: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 7
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7: FALSE (8.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled <= 210: TRUE (28.0/2.0)
|   |   |   |   |   |   |   |   |   |   redTotalMinionsKilled > 210
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 3
|   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 1
|   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 1: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 1: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAssists > 1: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 3
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 6
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 5: TRUE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 5: FALSE (14.0/3.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 6: TRUE (8.0/1.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE: TRUE (125.0/50.0)
|   |   |   |   |   |   |   redHeralds > 0
|   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   |   redAvgLevel > 6.6
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7
|   |   |   |   |   |   |   |   |   |   |   blueKills <= 7
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 970: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 970: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   blueKills > 7: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 1112: TRUE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 1112: FALSE (6.0)
|   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   blueKills <= 6: TRUE (12.0)
|   |   |   |   |   |   |   |   |   blueKills > 6
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 1371: FALSE (3.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 1371: TRUE (5.0/1.0)
|   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   blueTowersDestroyed <= 0
|   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   redAvgLevel <= 6.8: TRUE (54.0/21.0)
|   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   blueKills <= 3: FALSE (4.0)
|   |   |   |   |   |   |   |   |   blueKills > 3
|   |   |   |   |   |   |   |   |   |   redAssists <= 8
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 7: TRUE (19.0/4.0)
|   |   |   |   |   |   |   |   |   |   |   blueAvgLevel > 7
|   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 5
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists <= 5: TRUE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   blueAssists > 5
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 1320: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 1320: TRUE (2.0)
|   |   |   |   |   |   |   |   |   |   |   |   redAssists > 5: FALSE (4.0)
|   |   |   |   |   |   |   |   |   |   redAssists > 8: FALSE (3.0)
|   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   blueDeaths <= 5
|   |   |   |   |   |   |   |   |   blueGoldDiff <= 966: FALSE (15.0/2.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > 966
|   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 1006: TRUE (6.0)
|   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 1006: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   blueDeaths > 5
|   |   |   |   |   |   |   |   |   blueAssists <= 10
|   |   |   |   |   |   |   |   |   |   blueAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel <= 6.6: TRUE (3.0)
|   |   |   |   |   |   |   |   |   |   |   redAvgLevel > 6.6: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   blueAvgLevel > 6.8: TRUE (9.0)
|   |   |   |   |   |   |   |   |   blueAssists > 10: FALSE (2.0)
|   |   |   |   |   |   blueTowersDestroyed > 0
|   |   |   |   |   |   |   blueTowersDestroyed <= 1
|   |   |   |   |   |   |   |   redAvgLevel <= 6.8
|   |   |   |   |   |   |   |   |   redAssists <= 13: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   redAssists > 13: FALSE (2.0)
|   |   |   |   |   |   |   |   redAvgLevel > 6.8
|   |   |   |   |   |   |   |   |   blueGoldDiff <= 1477
|   |   |   |   |   |   |   |   |   |   redAssists <= 8: FALSE (8.0)
|   |   |   |   |   |   |   |   |   |   redAssists > 8
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff <= 428: FALSE (2.0)
|   |   |   |   |   |   |   |   |   |   |   blueGoldDiff > 428: TRUE (2.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > 1477: TRUE (2.0)
|   |   |   |   |   |   |   blueTowersDestroyed > 1: TRUE (2.0)
|   |   |   |   redAvgLevel > 7
|   |   |   |   |   redAssists <= 3
|   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   blueExperienceDiff <= -808: FALSE (3.0)
|   |   |   |   |   |   |   |   blueExperienceDiff > -808
|   |   |   |   |   |   |   |   |   blueKills <= 6
|   |   |   |   |   |   |   |   |   |   blueAssists <= 6
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 1091: TRUE (12.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 1091: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueAssists > 6: FALSE (4.0/1.0)
|   |   |   |   |   |   |   |   |   blueKills > 6: TRUE (4.0)
|   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   blueTotalMinionsKilled <= 241: FALSE (2.0)
|   |   |   |   |   |   |   |   blueTotalMinionsKilled > 241: TRUE (2.0)
|   |   |   |   |   |   redAvgLevel > 7.2: TRUE (4.0)
|   |   |   |   |   redAssists > 3
|   |   |   |   |   |   redHeralds <= 0
|   |   |   |   |   |   |   blueHeralds = FALSE
|   |   |   |   |   |   |   |   redAvgLevel <= 7.2
|   |   |   |   |   |   |   |   |   blueGoldDiff <= 293: TRUE (6.0)
|   |   |   |   |   |   |   |   |   blueGoldDiff > 293
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = TRUE
|   |   |   |   |   |   |   |   |   |   |   blueDeaths <= 4: TRUE (6.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   blueDeaths > 4
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 73: FALSE (11.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 73
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists <= 7: TRUE (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   redAssists > 7: FALSE (3.0/1.0)
|   |   |   |   |   |   |   |   |   |   blueFirstBlood = FALSE
|   |   |   |   |   |   |   |   |   |   |   redAssists <= 4: FALSE (6.0)
|   |   |   |   |   |   |   |   |   |   |   redAssists > 4
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff <= 220: FALSE (7.0)
|   |   |   |   |   |   |   |   |   |   |   |   blueExperienceDiff > 220: TRUE (7.0/1.0)
|   |   |   |   |   |   |   |   redAvgLevel > 7.2: TRUE (4.0/1.0)
|   |   |   |   |   |   |   blueHeralds = TRUE
|   |   |   |   |   |   |   |   redAssists <= 4: TRUE (3.0)
|   |   |   |   |   |   |   |   redAssists > 4: FALSE (26.0/5.0)
|   |   |   |   |   |   redHeralds > 0: FALSE (20.0/5.0)
|   blueGoldDiff > 1769: TRUE (2283.0/325.0)

Number of Leaves  : 	541

Size of the tree : 	1081


Time taken to build model: 0.49 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        6828               69.1163 %
Incorrectly Classified Instances      3051               30.8837 %
Kappa statistic                          0.3823
Mean absolute error                      0.3597
Root mean squared error                  0.5   
Relative absolute error                 71.947  %
Root relative squared error             99.997  %
Total Number of Instances             9879     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,698    0,316    0,689      0,698    0,694      0,382    0,693     0,646     FALSE
                 0,684    0,302    0,693      0,684    0,689      0,382    0,693     0,628     TRUE
Weighted Avg.    0,691    0,309    0,691      0,691    0,691      0,382    0,693     0,637     

=== Confusion Matrix ===

    a    b   <-- classified as
 3456 1493 |    a = FALSE
 1558 3372 |    b = TRUE
```
### DECISION STUMP
```sehll
=== Run information ===

Scheme:       weka.classifiers.trees.DecisionStump 
Relation:     datos_simplificados
Instances:    9879
Attributes:   23
              blueWins
              blueFirstBlood
              blueKills
              blueDeaths
              blueAssists
              blueDragons
              blueHeralds
              blueTowersDestroyed
              blueAvgLevel
              blueTotalMinionsKilled
              blueGoldDiff
              blueExperienceDiff
              redFirstBlood
              redKills
              redDeaths
              redAssists
              redDragons
              redHeralds
              redTowersDestroyed
              redAvgLevel
              redTotalMinionsKilled
              redGoldDiff
              redExperienceDiff
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

Decision Stump

Classifications

blueGoldDiff <= 212.0 : FALSE
blueGoldDiff > 212.0 : TRUE
blueGoldDiff is missing : FALSE

Class distributions

blueGoldDiff <= 212.0
FALSE	TRUE	
0.7116878196628149	0.2883121803371851	
blueGoldDiff > 212.0
FALSE	TRUE	
0.2591304347826087	0.7408695652173913	
blueGoldDiff is missing
FALSE	TRUE	
0.5009616357930965	0.4990383642069035	


Time taken to build model: 0.05 seconds

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


