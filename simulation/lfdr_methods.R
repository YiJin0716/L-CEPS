# Local-FDR comparators. Definitions are fixed before evaluating any test outcome.
# Sun & Cai (2007), Section 4.2, equation (10); Jin & Cai (2007), equation (21).
# Abraham, Castillo & Roquain (arXiv:2102.00929v3), equations (7)--(10).
# Miller & Breheny (2023), using the authors' ncvreg::local_mfdr implementation.

marginal_z <- function(X, y) {
  n <- nrow(X)
  X <- scale(X, center=TRUE, scale=FALSE)
  y <- y - mean(y)
  r <- as.vector(crossprod(X, y)) / sqrt(colSums(X^2) * sum(y^2))
  r <- pmin(pmax(r, -1 + 1e-15), 1 - 1e-15)
  t <- r * sqrt((n-2) / (1-r^2))
  # Use the survival probability to avoid rounding large positive CDFs to one.
  sign(t) * qnorm(pt(abs(t), df=n-2, lower.tail=FALSE), lower.tail=FALSE)
}

adaptz <- function(z) {
  p <- length(z)
  # Numerical evaluation of Jin--Cai (21), with theoretical null N(0,1).
  # gamma=.49, 101 equally spaced frequencies, trapezoidal integration in xi.
  xi <- seq(0, 1, length.out=101)
  quadrature <- rep(.01, 101); quadrature[c(1,101)] <- .005
  frequencies <- seq(0, sqrt(.98*log(p)), length.out=101)
  estimate <- vapply(frequencies, function(t) {
    characteristic <- rowMeans(cos(outer(t*xi, z)))
    2*sum(quadrature * (1-xi) * exp((t*xi)^2/2) * characteristic)
  }, numeric(1))
  pi0 <- pmin(1, pmax(0, min(estimate)))
  bandwidth <- bw.ucv(z)
  density_fit <- density(z, bw=bandwidth, kernel="gaussian", n=512)
  f <- approx(density_fit$x, density_fit$y, xout=z)$y
  list(score=pmin(1, pi0*dnorm(z)/pmax(f, .Machine$double.xmin)),
       pi0=pi0, bandwidth=bandwidth)
}

cumulative_ell <- function(z) {
  p <- length(z); z2 <- z^2
  g <- rep(.5 / sqrt(2*pi), p)
  positive <- z2 > 1e-12
  g[positive] <- -expm1(-z2[positive]/2) / z2[positive] / sqrt(2*pi)
  phi <- dnorm(z)
  # Concavity gives a unique interior optimum or a boundary maximum.
  derivative <- function(w) sum((g-phi) / ((1-w)*phi+w*g))
  lower <- 1/p
  w <- if (derivative(lower) <= 0) lower else if (derivative(1) >= 0) 1 else
    uniroot(derivative, c(lower,1), tol=1e-12)$root
  list(score=(1-w)*phi / ((1-w)*phi+w*g), weight=w)
}

evaluate_lfdr <- function(X, y, response, seed) {
  stopifnot(nrow(X)==length(y), all(is.finite(X)), all(is.finite(y)))
  p <- ncol(X); colnames(X) <- paste0("x",seq_len(p))
  z <- marginal_z(X, y)
  az <- adaptz(z); ell <- cumulative_ell(z)
  set.seed(seed)
  fit <- ncvreg::cv.ncvreg(X, y, family=if(response=="class") "binomial" else "gaussian",
                         penalty="lasso", nfolds=10, returnX=TRUE)
  local <- ncvreg::local_mfdr(fit$fit, lambda=fit$lambda.min, method="ashr")
  stopifnot(nrow(local)==p, identical(rownames(local), colnames(X)))
  scores <- cbind(adaptz=az$score, cumulative_ell=ell$score, lasso_mfdr=local$mfdr)
  stopifnot(all(is.finite(scores)), min(scores)>=0, max(scores)<=1)
  list(marginal_z=z, scores=scores, lasso_z=local$z,
       lasso_beta=local$Estimate, lasso_selected=local$Selected=="*",
       fold=fit$fold, lambda=fit$lambda.min, pi0=az$pi0,
       bandwidth=az$bandwidth, ell_weight=ell$weight)
}

# Persistent line-based worker. Input arrays are little-endian float64, X row-major then y.
if (sys.nframe()==0L) {
  suppressPackageStartupMessages(library(ncvreg))
  stopifnot(requireNamespace("ashr", quietly=TRUE), requireNamespace("jsonlite", quietly=TRUE))
  input <- file("stdin", open="r")
  repeat {
    line <- readLines(input, n=1, warn=FALSE)
    if (!length(line)) break
    request <- jsonlite::fromJSON(line)
    warning_messages <- character()
    started <- proc.time()[3]
    value <- tryCatch(withCallingHandlers({
      connection <- file(request$input, "rb")
      values <- readBin(connection, what="double", n=request$n*request$p+request$n,
                        size=8, endian="little")
      close(connection)
      stopifnot(length(values)==request$n*request$p+request$n)
      X <- matrix(values[seq_len(request$n*request$p)], nrow=request$n, byrow=TRUE)
      y <- tail(values, request$n)
      out <- evaluate_lfdr(X,y,request$response,request$seed)
      c(list(ok=TRUE), out)
    }, warning=function(w) {
      warning_messages <<- c(warning_messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }), error=function(e) list(ok=FALSE,error=conditionMessage(e)))
    value$warnings <- unique(warning_messages)
    value$seconds <- unname(proc.time()[3]-started)
    cat(jsonlite::toJSON(value, digits=17, auto_unbox=TRUE, na="null"), "\n", sep="")
    flush.console()
  }
  close(input)
}
