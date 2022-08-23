library(data.table)

########################################################
# gsod - 572a71e434c9977a3a3434c7 - ENET
########################################################
x <- list(
  data.table(
    name="PDM2",
    fit_RAM=5958309000,
    trans_RAM=8993776000
  ),
  data.table(
    name="PNI",
    fit_RAM=22310490000,
    trans_RAM=26682509000
  ),
  data.table(
    name="ST",
    fit_RAM=41858751000,
    trans_RAM=36833603000
  ),
  data.table(
    name="ENETCD",
    fit_RAM=38408001000,
    trans_RAM=33034676000
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x
x[,sum(fit_RAM)]
x[,sum(trans_RAM)]

########################################################
# gsod - 572a720934c9977a3a3434d5 - GLMCD or linear reg
########################################################
x <- list(
  data.table(
    name="PDM2",
    fit_RAM=5962248000,
    trans_RAM=8993835000
  ),
  data.table(
    name="PNI",
    fit_RAM=22310490000,
    trans_RAM=26682438000
  ),
  data.table(
    name="ST",
    fit_RAM=42443348000,
    trans_RAM=36833530000
  ),
  data.table(
    name="GLMCD",
    fit_RAM=66719200000,
    trans_RAM=33033399000
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x
x[,sum(fit_RAM)]
x[,sum(trans_RAM)]

########################################################
# gsod - 572a73fd34c9977a3a3434ff - SGD
########################################################
x <- list(
  data.table(
    name="PDM2",
    fit_RAM=5962248000,
    trans_RAM=8728989000
  ),
  data.table(
    name="PNI",
    fit_RAM=22310490000,
    trans_RAM=26682616000
  ),
  data.table(
    name="ST",
    fit_RAM=42443348000,
    trans_RAM=36833816000
  ),
  data.table(
    name="SGDRA",
    fit_RAM=66723175000,
    trans_RAM=23570234000
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x
x[,sum(fit_RAM)]
x[,sum(trans_RAM)]

########################################################
# gsod - 572a720634c9977a3a3434d1 - Ridge 1
########################################################
x <- list(
  data.table(
    name="PDM2",
    fit_RAM=5962754000,
    trans_RAM=8993826000
  ),
  data.table(
    name="PNI",
    fit_RAM=22310490000,
    trans_RAM=26682444000
  ),
  data.table(
    name="ST",
    fit_RAM=42443348000,
    trans_RAM=42443348000
  ),
  data.table(
    name="ENETCD",
    fit_RAM=38408025000,
    trans_RAM=33569520000
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x
x[,sum(fit_RAM)]
x[,sum(trans_RAM)]

########################################################
# gsod - 572a720434c9977a3a3434cd - Ridge 2
########################################################
x <- list(
  data.table(
    name="PDM2",
    fit_RAM=5876746000,
    trans_RAM=8992021000
  ),
  data.table(
    name="PNI",
    fit_RAM=22310490000,
    trans_RAM=26682291000
  ),
  data.table(
    name="ST",
    fit_RAM=41858751000,
    trans_RAM=36833386000
  ),
  data.table(
    name="ENETCD",
    fit_RAM=66719068000,
    trans_RAM=33569105000
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x
x[,sum(fit_RAM)]
x[,sum(trans_RAM)]

########################################################
# gsod - 572b452d34c9977a3a343535 - DIFF3 + ES XGB
########################################################

x <- list(
  data.table(
    name="ORDCAT2",
    fit_RAM=5296017000,
    trans_RAM=6966002000,
    fit_time=7.471280813217163,
    trans_time=24.195943117141724
  ),
  data.table(
    name="PCCAT",
    fit_RAM=7037093000,
    trans_RAM=5466355000,
    fit_time=5.6853721141815186,
    trans_time=0
  ),
  data.table(
    name="PNIA",
    fit_RAM=22777413000,
    trans_RAM=26282155000,
    fit_time=1.6136479377746582,
    trans_time=13.380331039428711
  ),
  data.table(
    name="DIFF3",
    fit_RAM=24810661000,
    trans_RAM=21887064000,
    fit_time=212.19247698783875,
    trans_time=5.031401872634888
  ),
  data.table(
    name="BIND",
    fit_RAM=26134911000,
    trans_RAM=26134911000,
    fit_time=0,
    trans_time=0
  ),
  data.table(
    name="ESXGBR",
    fit_RAM=44353640000,
    trans_RAM=34655994000,
    fit_time=96434.85916996002,
    trans_time=2660.5794880390167
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x[, fit_time := round(fit_time / 60, 1)]
x[, trans_time := round(trans_time / 60, 1)]
x

########################################################
# AirBnB - 572bf65d34c997f3a7918593 - Ridge - all vars
########################################################

x <- list(
  data.table(
    name="PDM2",
    fit_RAM=1222788000,
    trans_RAM=2241444000,
    fit_time=69.11352014541626,
    trans_time=77.56121397018433
  ),
  data.table(
    name="PTM2",
    fit_RAM=25895468000,
    trans_RAM=55293854000,
    fit_time=1570.6677269935608,
    trans_time=3599.9554159641266
  ),
  data.table(
    name="PNI",
    fit_RAM=24186378000,
    trans_RAM=23364799000,
    fit_time=4.740816116333008,
    trans_time=1.5819499492645264
  ),
  data.table(
    name="ST",
    fit_RAM=23364799000,
    trans_RAM=23102680000,
    fit_time=1.5336599349975586,
    trans_time=0
  ),
  data.table(
    name="ENETCD",
    fit_RAM=41745021000,
    trans_RAM=41519952000,
    fit_time=398.7842299938202,
    trans_time=55.03802180290222
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x[, fit_time := round(fit_time / 60, 1)]
x[, trans_time := round(trans_time / 60, 1)]
x

########################################################
# AirBnB - 572bf66234c997f3a791859e - Ridge - words only
########################################################

x <- list(
  data.table(
    name="PDM2",
    fit_RAM=25895468000,
    trans_RAM=54809260000,
    fit_time=1570.6677269935608,
    trans_time=3187.544184923172
  ),
  data.table(
    name="ENETCDWC",
    fit_RAM=38778147000,
    trans_RAM=38778147000,
    fit_time=188.7876329421997,
    trans_time=40.69124221801758
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x[, fit_time := round(fit_time / 60, 1)]
x[, trans_time := round(trans_time / 60, 1)]
x



########################################################
# AirBnB - 575433c834c9970b8d707094 - ridge - 10_0_06 float32
########################################################
db.leaderboard.findOne({'_id': ObjectId('575433c834c9970b8d707094')}, {'task_info': 1})
x <- list(
  data.table(
    name="PDM2",
    fit_RAM=,
    trans_RAM=,
    fit_time=,
    trans_time=
  ),
  data.table(
    name="ENETCDWC",
    fit_RAM=,
    trans_RAM=,
    fit_time=,
    trans_time=
  )
)
x <- rbindlist(x)
x[, fit_RAM := fit_RAM / 1e9]
x[, trans_RAM := trans_RAM / 1e9]
x[, fit_time := round(fit_time / 60, 1)]
x[, trans_time := round(trans_time / 60, 1)]
x

########################################################
# AirBnB - 5751a9e334c99749ad93bf8d - blender - 10_0_06 float32
########################################################
