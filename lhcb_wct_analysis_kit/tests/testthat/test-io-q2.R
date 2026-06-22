test_that("q2 reconstruction from muon four-vectors matches the closed form", {
  df <- data.frame(
    muplus_PX = 1000, muplus_PY = 200, muplus_PZ = 5000, muplus_PE = 5200,
    muminus_PX = -800, muminus_PY = 150, muminus_PZ = 4000, muminus_PE = 4150)
  q2 <- reconstruct_q2(df)
  E <- 5200 + 4150; px <- 1000 - 800; py <- 200 + 150; pz <- 5000 + 4000
  expect_equal(q2, (E*E - px*px - py*py - pz*pz)/1e6)
})

test_that("MeV^2 -> GeV^2 conversion divides by 1e6", {
  df <- data.frame(muplus_PX=0,muplus_PY=0,muplus_PZ=0,muplus_PE=3000,
                   muminus_PX=0,muminus_PY=0,muminus_PZ=0,muminus_PE=2000)
  expect_equal(reconstruct_q2(df), (5000^2)/1e6)
})

test_that("missing q2 branches raise a clear error", {
  expect_error(reconstruct_q2(data.frame(a=1)), "missing")
})

test_that("K* mass reconstruction from K+ pi- four-vectors is real and >= 0", {
  df <- data.frame(Kplus_PX=600,Kplus_PY=10,Kplus_PZ=2000,Kplus_PE=2100,
                   piminus_PX=-100,piminus_PY=20,piminus_PZ=900,piminus_PE=920)
  m <- reconstruct_kst_mass(df)
  expect_true(is.finite(m) && m >= 0)
})

test_that("B0 and K* selection windows include/exclude correctly", {
  bm <- c(5100, 5230, 5280, 5330, 5400)
  km <- c(700, 795.9, 900, 995.9, 1100)
  keep <- (bm >= B0_M_MIN & bm <= B0_M_MAX) & (km >= KST_M_MIN & km <= KST_M_MAX)
  expect_equal(keep, c(FALSE, TRUE, TRUE, TRUE, FALSE))
})

test_that("file_sha256 is stable for a known byte string", {
  tf <- tempfile(); writeLines("abc", tf)
  h1 <- file_sha256(tf); h2 <- file_sha256(tf)
  expect_identical(h1, h2)
  expect_match(h1, "^[0-9a-f]{64}$")
})
