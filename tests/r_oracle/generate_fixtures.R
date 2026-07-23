# Offline, dev-time R-oracle fixture generator.
#
# Sources the ORIGINAL R backend and dumps reference outputs for the sample
# data into tests/r_oracle/fixtures/*.csv, which the Python test suite
# compares against. Run from this directory, in the OpenLDMTest environment:
#
#   cd OpenLDMPy/tests/r_oracle
#   mamba run -n OpenLDMTest Rscript generate_fixtures.R
#
# R is never invoked at application runtime or test time — only this script.

source("../../../src/Rasterise_dev_68akj.r")

data_dir <- "../../data/example"
fix_dir <- "fixtures"
dir.create(fix_dir, showWarnings = FALSE)

T1File <- file.path(data_dir, "LULC/1985.tif")
T2File <- file.path(data_dir, "LULC/1995.tif")
T3File <- file.path(data_dir, "LULC/2005.tif")
MaskFile <- file.path(data_dir, "MaksFiles/Mask/Mask2005.tif")
AoiFile <- file.path(data_dir, "MaksFiles/AOI/BigCities2005.tif")

mydemand <- as.numeric(c(1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992))

# --- 1. Observed transition matrix (createTM) --------------------------------
TM <- createTM(T1File, T2File)
write.table(TM, file.path(fix_dir, "tm_1985_1995.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)

# --- 2. Markov-projected future matrix (getNewTM, no demand) -----------------
newTM <- getNewTM(TM)
write.table(newTM, file.path(fix_dir, "newtm_markov.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)

# --- 3. Demand-constrained future matrix (getNewTM with demand) --------------
newTMd <- getNewTM(TM, mydemand)
write.table(newTMd, file.path(fix_dir, "newtm_demand.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)

# --- 4. Yearly step matrix (getYearlyMatrix; needs expm) ---------------------
ok <- tryCatch({
  yearly <- getYearlyMatrix(newTMd, 2, 1)
  write.table(yearly, file.path(fix_dir, "yearly_from_newtm_demand.csv"),
              sep = ",", row.names = FALSE, col.names = FALSE)
  TRUE
}, error = function(e) { print(e); FALSE })
if (!ok) print("SKIPPED: getYearlyMatrix (expm not installed?)")

# --- 5. Kappa statistics of the 1985->1995 matrix (kappa) --------------------
k <- kappa(TM)
kf <- data.frame(
  name = c("sum.n", "sum.naive", "sum.var", "sum.kappa", "sum.kvar"),
  value = c(k$sum.n, k$sum.naive, k$sum.var, k$sum.kappa, k$sum.kvar)
)
write.csv(kf, file.path(fix_dir, "kappa_scalars.csv"), row.names = FALSE)
kpc <- data.frame(
  user.naive = k$user.naive, prod.naive = k$prod.naive,
  user.kappa = k$user.kappa, user.kvar = k$user.kvar,
  prod.kappa = k$prod.kappa, prod.kvar = k$prod.kvar
)
write.csv(kpc, file.path(fix_dir, "kappa_per_class.csv"), row.names = FALSE)

# PyKappasummary machine string, for byte-comparison.
writeLines(PyKappasummary(k), file.path(fix_dir, "py_kappa_summary.txt"))

# --- 6. Neighbour weights (ComputeNW, the worker the parallel path uses) -----
for (i in c(1, 9)) {
  nw <- ComputeNW(i, T2File, withNA = NA, wSize = NA)
  write.csv(nw, file.path(fix_dir, sprintf("nw_class%d_nowin.csv", i)),
            row.names = FALSE)
  nww <- ComputeNW(i, T2File, withNA = NA, wSize = 3)
  write.csv(nww, file.path(fix_dir, sprintf("nw_class%d_win3.csv", i)),
            row.names = FALSE)
}

# --- 7. Raster-format mask/AOI clipping (getMasked) --------------------------
ok <- tryCatch({
  m <- getMasked(T2File, maskFile = MaskFile)
  v <- m[[1]][]
  write.csv(data.frame(id = which(!is.na(v)), value = v[!is.na(v)]),
            file.path(fix_dir, "masked_t2_values.csv"), row.names = FALSE)
  a <- getMasked(T2File, aoiFile = AoiFile)
  va <- a[[1]][]
  write.csv(data.frame(id = which(!is.na(va)), value = va[!is.na(va)]),
            file.path(fix_dir, "aoi_t2_values.csv"), row.names = FALSE)
  TRUE
}, error = function(e) { print(e); FALSE })
if (!ok) print("SKIPPED: getMasked fixtures")

# --- 8. Logistic (glm) suitability for class 1, base R only ------------------
# Fit exactly as fitModelSeparately does for modelType='logistic', then
# predict on the T2 drivers (the INTENDED semantics).
drvs85 <- c(
  DistanceToDrainage = file.path(data_dir, "Drivers/drivers_85/dist_stream.img"),
  DistanceToBuiltup = file.path(data_dir, "Drivers/drivers_85/Dist_urban.img"),
  DistanceToRoad = file.path(data_dir, "Drivers/drivers_85/road_final.img"),
  Elevation = file.path(data_dir, "Drivers/commonDrivers/elevation.img")
)
drvs95 <- c(
  DistanceToDrainage = file.path(data_dir, "Drivers/drivers_95/dist_stream.img"),
  DistanceToBuiltup = file.path(data_dir, "Drivers/drivers_95/Dist_urban.img"),
  DistanceToRoad = file.path(data_dir, "Drivers/drivers_95/road_final.img"),
  Elevation = file.path(data_dir, "Drivers/commonDrivers/elevation.img")
)
ok <- tryCatch({
  dT1 <- getDataTable(T1File, TRUE, NA)
  drvT1 <- getDataTable(drvs85, FALSE, NA, names(drvs85))
  drvT2 <- getDataTable(drvs95, FALSE, NA, names(drvs95))
  dta <- data.table::data.table(cbind(id = seq(1, dim(dT1)[1]), T1 = dT1, TD1 = drvT1))
  dta <- dta[rowSums(dT1, na.rm = TRUE) != 0]
  dta[is.na(dta)] <- 0
  frml <- paste(colnames(dta)[2], "~",
                paste(colnames(dta)[(2 + dim(dT1)[2]):(1 + dim(dT1)[2] + length(drvs85))],
                      collapse = "+"))
  fit <- glm(formula(frml), family = binomial, data = dta, na.action = "na.omit")
  write.csv(data.frame(term = names(coef(fit)), estimate = as.numeric(coef(fit))),
            file.path(fix_dir, "glm_class1_coefs.csv"), row.names = FALSE)

  pred_dt <- data.table::data.table(cbind(id = seq(1, dim(drvT2)[1]), TD1 = drvT2))
  pred_dt <- pred_dt[rowSums(drvT2, na.rm = TRUE) != 0]
  probs <- predict(fit, newdata = pred_dt, type = "response")
  write.csv(data.frame(id = pred_dt$id, weight = as.numeric(probs)),
            file.path(fix_dir, "glm_class1_suitability_t2.csv"), row.names = FALSE)
  TRUE
}, error = function(e) { print(e); FALSE })
if (!ok) print("SKIPPED: glm suitability fixture")

# --- 9. Pontius three-map disagreement decomposition (kappa.agreementindex)
# -------------------------------------------------------------------------
ok <- tryCatch({
  ki <- kappa.agreementindex(simulationFile = T3File, actualFile = T2File, baseFile = T1File)
  write.table(ki$propertionmatrix, file.path(fix_dir, "pontius_proportionmatrix.csv"),
              sep = ",", row.names = FALSE, col.names = FALSE)

  overall_names <- c(
    "kstandard.overall", "kno.overall", "klocation.overall", "kallocation.overall",
    "khistogram.overall", "kquantity.overall", "ksimulation.overall",
    "ktransition.overall", "ktranslocation.overall",
    "allocationdisagreemnt.overall", "allocationagreemnt.overall", "shift.overall",
    "exchange.overall", "quantitydisagreemnt.overall", "cumulativedisagreemnt.overall"
  )
  write.csv(
    data.frame(name = overall_names, value = as.numeric(ki[overall_names])),
    file.path(fix_dir, "pontius_overall.csv"), row.names = FALSE
  )

  classwise_names <- c(
    "kstandard.classwise", "kno.classwise", "klocation.classwise", "kallocation.classwise",
    "khistogram.classwise", "kquantity.classwise", "ksimulation.classwise",
    "ktransition.classwise", "ktranslocation.classwise",
    "allocationdisagreemnt.classwise", "allocationagreemnt.classwise", "shift.classwise",
    "exchange.classwise", "quantitydisagreemnt.classwise", "cumulativedisagreemnt.classwise"
  )
  cw <- as.data.frame(ki[classwise_names])
  colnames(cw) <- classwise_names
  write.csv(cw, file.path(fix_dir, "pontius_classwise.csv"), row.names = FALSE)
  TRUE
}, error = function(e) { print(e); FALSE })
if (!ok) print("SKIPPED: kappa.agreementindex (Pontius) fixtures")

print("Fixture generation finished.")
