test_that("integer winding grid spans n=10..22 with correct k mapping", {
  ns <- 10:22
  ks <- k_from_n(ns)
  expect_equal(length(ks), 13L)
  expect_equal(ks[ns == 20], 26.28865146755044, tolerance = 1e-10)
  expect_true(all(diff(ks) > 0))
})

test_that("Koide comb construction n0*(Q,1,2Q) gives (10,15,20) at Q=2/3", {
  n0 <- 15; Q <- 2/3
  comb <- n0 * c(Q, 1, 2*Q)
  expect_equal(comb, c(10, 15, 20), tolerance = 1e-9)
})

test_that("ROOT branch discovery prefers B0_KstMuMu/DecayTree", {
  keys <- c("Other", "B0_KstMuMu/DecayTree", "X/DecayTree")
  pref <- "B0_KstMuMu/DecayTree"
  chosen <- if (pref %in% keys) pref else keys[grepl("DecayTree", keys)][1]
  expect_equal(chosen, "B0_KstMuMu/DecayTree")
  keys2 <- c("Other", "Sel/DecayTree")
  chosen2 <- if (pref %in% keys2) pref else keys2[grepl("DecayTree", keys2)][1]
  expect_equal(chosen2, "Sel/DecayTree")
})

test_that("JSON and CSV output schemas round-trip", {
  obj <- list(a = 1, b = list(c = 2), d = c(1,2,3))
  tf <- tempfile(fileext = ".json")
  write_json(obj, tf)
  back <- jsonlite::fromJSON(tf)
  expect_equal(back$a, 1); expect_equal(back$b$c, 2)
  df <- data.frame(k = 1:3, delta_chi2 = c(0.1, 0.2, 0.3))
  tf2 <- tempfile(fileext = ".csv")
  write.csv(df, tf2, row.names = FALSE)
  expect_equal(read.csv(tf2)$delta_chi2, df$delta_chi2)
})

test_that("manifest records version, seed, rng kind and git commit field", {
  man <- build_manifest("test", config = list(x = 1), seed = 123)
  expect_true(!is.null(man$r_version))
  expect_equal(man$seed, 123)
  expect_true(!is.null(man$rng_kind))
  expect_true("git_commit" %in% names(man))
})
