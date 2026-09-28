#!/usr/bin/env Rscript

# Fit the plan-specified group-based multitrajectory models on the strict
# 1,458-stay cohort.  GE3 is the primary trajectory set and FULL5 is the
# five-day complete-case sensitivity set.  The script intentionally does not
# rebuild the cohort or impute the marker table before fitting.
#
# The Python preparation stage already performed pooled, marker-specific
# winsorization and z-scoring.  Therefore gbmt::gbmt() is called with
# scaling=0; using its default scaling=2 would standardize within each stay and
# would change the prespecified analysis.

options(stringsAsFactors = FALSE)

parse_cli <- function(args) {
  out <- list(
    input_dir = NULL,
    output_dir = NULL,
    sets = c("GE3", "FULL5"),
    k_min = 2L,
    k_max = 6L,
    nstart = 10L,
    maxit = 1000L,
    tol = 1e-4,
    seed = 20260914L
  )
  if (length(args) == 0L) return(out)
  if (length(args) %% 2L != 0L) {
    stop("Arguments must be supplied as --name value pairs.", call. = FALSE)
  }
  for (i in seq(1L, length(args), by = 2L)) {
    key <- sub("^--", "", args[[i]])
    value <- args[[i + 1L]]
    if (!(key %in% names(out))) stop("Unknown argument: --", key, call. = FALSE)
    if (key == "sets") {
      out[[key]] <- strsplit(value, ",", fixed = TRUE)[[1L]]
    } else if (key %in% c("k_min", "k_max", "nstart", "maxit", "seed")) {
      out[[key]] <- as.integer(value)
    } else if (key == "tol") {
      out[[key]] <- as.numeric(value)
    } else {
      out[[key]] <- value
    }
  }
  out
}

args <- parse_cli(commandArgs(trailingOnly = TRUE))
if (is.null(args$input_dir) || is.null(args$output_dir)) {
  stop(
    paste(
      "Usage:",
      "Rscript fit_adhf_gbmt_grid_v1.R",
      "--input_dir <trajectory_model_inputs_v1_1458>",
      "--output_dir <gbmt_results_v1_1458>",
      "[--sets GE3,FULL5] [--k_min 2] [--k_max 6]",
      "[--nstart 10] [--maxit 1000] [--tol 1e-4] [--seed 20260914]"
    ),
    call. = FALSE
  )
}

if (!requireNamespace("gbmt", quietly = TRUE)) {
  stop(
    "The gbmt package is required. Install it once with install.packages('gbmt').",
    call. = FALSE
  )
}

input_dir <- normalizePath(path.expand(args$input_dir), mustWork = TRUE)
output_dir <- path.expand(args$output_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (length(args$sets) == 0L || any(!args$sets %in% c("GE2", "GE3", "FULL5"))) {
  stop("--sets must contain only GE2, GE3, and/or FULL5.", call. = FALSE)
}
if (args$k_min < 2L || args$k_max < args$k_min) {
  stop("Require 2 <= k_min <= k_max.", call. = FALSE)
}
if (args$nstart < 1L || args$maxit < 1L || args$tol <= 0) {
  stop("nstart, maxit, and tol must be positive.", call. = FALSE)
}

id_cols <- c("subject_id", "hadm_id", "stay_id")
markers <- c(
  "cumulative_net_fluid_balance_z",
  "urine_output_z",
  "creatinine_z",
  "iv_loop_dose_z"
)

stop_missing <- function(data, needed, label) {
  missing <- setdiff(needed, names(data))
  if (length(missing) > 0L) {
    stop(label, " is missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
}

rbind_fill <- function(frames) {
  if (length(frames) == 0L) return(data.frame())
  columns <- unique(unlist(lapply(frames, names)))
  aligned <- lapply(frames, function(frame) {
    absent <- setdiff(columns, names(frame))
    for (column in absent) frame[[column]] <- NA
    frame[, columns, drop = FALSE]
  })
  out <- do.call(rbind, aligned)
  rownames(out) <- NULL
  out
}

read_model_data <- function(set_name) {
  path <- file.path(
    input_dir,
    paste0("trajectory_long_scaled_", set_name, "_v1.csv")
  )
  if (!file.exists(path)) stop("Input file not found: ", path, call. = FALSE)
  long <- read.csv(path, check.names = FALSE, na.strings = c("", "NA"))
  stop_missing(
    long,
    c(id_cols, "day_num", "marker", "value_z"),
    basename(path)
  )
  long$marker <- as.character(long$marker)
  long$day_num <- suppressWarnings(as.numeric(long$day_num))
  long$value_z <- suppressWarnings(as.numeric(long$value_z))
  if (anyNA(long$day_num) || any(!long$day_num %in% 0:4)) {
    stop(basename(path), " contains invalid day_num values.", call. = FALSE)
  }
  if (any(!long$marker %in% markers)) {
    stop(
      basename(path), " contains unexpected marker names: ",
      paste(setdiff(unique(long$marker), markers), collapse = ", "),
      call. = FALSE
    )
  }
  key <- interaction(long$stay_id, long$day_num, long$marker, drop = TRUE, sep = "|")
  if (anyDuplicated(key)) {
    stop(basename(path), " has duplicate stay-day-marker records.", call. = FALSE)
  }

  # reshape() produces one row per stay-day and one numeric column per marker.
  wide <- reshape(
    long[, c(id_cols, "day_num", "marker", "value_z")],
    idvar = c(id_cols, "day_num"),
    timevar = "marker",
    direction = "wide"
  )
  names(wide) <- sub("^value_z\\.", "", names(wide))
  stop_missing(wide, c(id_cols, "day_num", markers), "reshaped trajectory data")
  wide$unit_id <- paste(wide$subject_id, wide$hadm_id, wide$stay_id, sep = "_")
  wide$unit_id <- as.character(wide$unit_id)
  wide$time <- as.numeric(wide$day_num)
  wide <- wide[order(wide$unit_id, wide$time), ]
  rownames(wide) <- NULL

  if (anyDuplicated(wide[, c("unit_id", "time")])) {
    stop("The reshaped data have duplicated unit-time records.", call. = FALSE)
  }
  units <- unique(wide$unit_id)
  n_days <- table(wide$unit_id)
  if (any(n_days != 5L)) {
    stop("Each model stay must retain five day rows (0--4).", call. = FALSE)
  }
  if (any(vapply(wide[markers], function(x) all(is.na(x)), logical(1)))) {
    stop("At least one marker is completely missing in ", set_name, ".", call. = FALSE)
  }

  id_map <- unique(wide[, c("unit_id", id_cols)])
  if (anyDuplicated(id_map$unit_id)) stop("unit_id is not unique.", call. = FALSE)
  list(
    long = long,
    wide = wide,
    data = wide[, c("unit_id", "time", markers)],
    id_map = id_map,
    n_units = length(units)
  )
}

normalised_posterior <- function(fit, k) {
  post <- as.matrix(fit$posterior)
  if (ncol(post) != k) {
    stop("gbmt posterior has ", ncol(post), " columns; expected ", k, ".", call. = FALSE)
  }
  post <- apply(post, 2L, as.numeric)
  post <- as.matrix(post)
  rs <- rowSums(post)
  if (any(!is.finite(rs)) || any(rs <= 0)) {
    stop("gbmt returned invalid posterior row sums.", call. = FALSE)
  }
  post / rs
}

fit_metrics <- function(fit, set_name, k, data_obj) {
  post <- normalised_posterior(fit, k)
  assignment <- max.col(post, ties.method = "first")
  # gbmt$assign should agree with modal posterior assignment.  Use modal
  # assignment for a stable output even if an older package version omits it.
  if (!is.null(fit$assign) && length(fit$assign) == nrow(post)) {
    fit_assignment <- suppressWarnings(as.integer(fit$assign))
    if (all(is.finite(fit_assignment)) && all(fit_assignment %in% seq_len(k))) {
      assignment <- fit_assignment
    }
  }
  counts <- tabulate(assignment, nbins = k)
  props <- counts / nrow(post)
  entropy <- 1 + sum(post * log(pmax(post, .Machine$double.xmin))) /
    (nrow(post) * log(k))
  appa <- vapply(seq_len(k), function(g) {
    if (counts[g] == 0L) return(NA_real_)
    mean(post[assignment == g, g], na.rm = TRUE)
  }, numeric(1))
  bic <- NA_real_
  if (!is.null(fit$ic)) {
    if (!is.null(names(fit$ic)) && "bic" %in% names(fit$ic)) {
      bic <- as.numeric(fit$ic[["bic"]])
    } else if (length(fit$ic) >= 2L) {
      bic <- as.numeric(fit$ic[[2L]])
    }
  }
  loglik <- suppressWarnings(as.numeric(fit$logLik))
  npar <- suppressWarnings(as.numeric(fit$npar))
  em <- as.data.frame(fit$em)
  converged <- NA
  n_iter <- NA_real_
  if ("converged" %in% names(em)) converged <- as.integer(tail(em$converged, 1L))
  if ("n.iter" %in% names(em)) n_iter <- as.numeric(tail(em$n.iter, 1L))

  # fit$data.orig is ordered by gbmt using the unique unit IDs in the data.
  fit_unit_order <- unique(as.character(fit$data.orig$unit_id))
  if (length(fit_unit_order) != nrow(post)) {
    stop("Could not align gbmt posterior rows to model stays.", call. = FALSE)
  }
  assign_df <- data.frame(
    unit_id = fit_unit_order,
    assigned_class = assignment,
    max_posterior = apply(post, 1L, max),
    stringsAsFactors = FALSE
  )
  colnames(post) <- paste0("posterior_class_", seq_len(k))
  posterior_df <- data.frame(unit_id = fit_unit_order, post, check.names = FALSE)
  assign_df <- merge(assign_df, posterior_df, by = "unit_id", sort = FALSE)
  assign_df <- merge(assign_df, data_obj$id_map, by = "unit_id", sort = FALSE)
  assign_df <- assign_df[order(assign_df$stay_id), ]
  rownames(assign_df) <- NULL

  summary_df <- data.frame(
    set = set_name,
    K = k,
    n_stays = nrow(post),
    logLik = loglik,
    n_parameters = npar,
    AIC = if (!is.null(fit$ic) && length(fit$ic) >= 1L) as.numeric(fit$ic[[1L]]) else NA_real_,
    BIC = bic,
    entropy = as.numeric(entropy),
    min_class_proportion = min(props),
    min_APPA = min(appa, na.rm = TRUE),
    converged = converged,
    n_iter = n_iter,
    stringsAsFactors = FALSE
  )
  for (g in seq_len(k)) {
    summary_df[[paste0("class_", g, "_n")]] <- counts[g]
    summary_df[[paste0("class_", g, "_proportion")]] <- props[g]
    summary_df[[paste0("class_", g, "_APPA")]] <- appa[g]
  }
  min_appa <- if (all(is.na(appa))) NA_real_ else min(appa, na.rm = TRUE)
  summary_df$min_APPA <- min_appa
  list(summary = summary_df, assignments = assign_df, posterior = post)
}

tidy_fitted <- function(fit, set_name, k, markers) {
  rows <- list()
  for (g in seq_len(k)) {
    mat <- as.data.frame(fit$fitted[[g]], check.names = FALSE)
    time <- suppressWarnings(as.numeric(rownames(mat)))
    if (length(time) != nrow(mat) || anyNA(time)) time <- seq_len(nrow(mat)) - 1
    mat$time <- time
    mat$class <- g
    mat$set <- set_name
    mat$K <- k
    mat <- mat[, c("set", "K", "class", "time", markers), drop = FALSE]
    rows[[g]] <- mat
  }
  do.call(rbind, rows)
}

write_plot <- function(fit, path, markers) {
  # The package's plot method draws estimated trajectories and uncertainty
  # bands.  A failure to draw a diagnostic plot must not discard a successful
  # model fit, so the error is recorded in a sidecar text file.
  ok <- TRUE
  msg <- character()
  tryCatch({
    grDevices::pdf(path, width = 12, height = 8, useDingbats = FALSE)
    on.exit(grDevices::dev.off(), add = TRUE)
    plot(
      fit,
      x.names = markers,
      bands = TRUE,
      observed = FALSE,
      equal.scale = FALSE,
      add.grid = TRUE,
      transparency = 80,
      add.legend = TRUE,
      titles = c("Cumulative net fluid balance (z)", "Urine output (z)",
                 "Creatinine (z)", "IV loop dose (z)"),
      xlab = "ICU-relative day"
    )
  }, error = function(e) {
    ok <<- FALSE
    msg <<- conditionMessage(e)
  })
  if (!ok) writeLines(msg, paste0(path, ".error.txt"))
  invisible(ok)
}

all_grid <- list()
all_set_counts <- list()
all_failures <- list()

for (set_name in args$sets) {
  message("Preparing ", set_name, "...")
  data_obj <- read_model_data(set_name)
  set_dir <- file.path(output_dir, set_name)
  dir.create(set_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(data_obj$data, file.path(set_dir, paste0("gbmt_input_data_", set_name, ".rds")))
  write.csv(data_obj$data, file.path(set_dir, paste0("gbmt_input_data_", set_name, ".csv")), row.names = FALSE)

  fits <- list()
  set_grid <- list()
  for (k in seq.int(args$k_min, args$k_max)) {
    message("  Fitting ", set_name, " K=", k, " ...")
    fit_path <- file.path(set_dir, paste0("gbmt_", set_name, "_K", k, ".rds"))
    seed_k <- as.integer(args$seed + 1000L * match(set_name, args$sets) + k)
    result <- tryCatch({
      set.seed(seed_k)
      fit <- gbmt::gbmt(
        x.names = markers,
        unit = "unit_id",
        time = "time",
        ng = k,
        d = 2,
        data = data_obj$data,
        scaling = 0,
        pruning = FALSE,
        nstart = args$nstart,
        tol = args$tol,
        maxit = args$maxit,
        quiet = TRUE
      )
      saveRDS(fit, fit_path)
      metrics <- fit_metrics(fit, set_name, k, data_obj)
      write.csv(
        metrics$assignments,
        file.path(set_dir, paste0("gbmt_", set_name, "_K", k, "_assignments.csv")),
        row.names = FALSE
      )
      fitted <- tidy_fitted(fit, set_name, k, markers)
      write.csv(
        fitted,
        file.path(set_dir, paste0("gbmt_", set_name, "_K", k, "_fitted_trajectories.csv")),
        row.names = FALSE
      )
      write_plot(
        fit,
        file.path(set_dir, paste0("gbmt_", set_name, "_K", k, "_trajectories.pdf")),
        markers
      )
      list(summary = metrics$summary, ok = TRUE)
    }, error = function(e) {
      failure <- data.frame(
        set = set_name,
        K = k,
        error = conditionMessage(e),
        stringsAsFactors = FALSE
      )
      all_failures[[length(all_failures) + 1L]] <<- failure
      list(summary = NULL, ok = FALSE)
    })
    if (isTRUE(result$ok)) {
      fits[[as.character(k)]] <- readRDS(fit_path)
      set_grid[[as.character(k)]] <- result$summary
    }
  }

  if (length(set_grid) == 0L) {
    warning("No successful gbmt model for ", set_name, ".")
    next
  }
  grid <- rbind_fill(set_grid)
  grid <- grid[order(grid$K), ]
  grid$bic_lower_than_previous <- NA
  if (nrow(grid) > 1L) {
    for (i in 2:nrow(grid)) {
      previous <- grid$BIC[match(grid$K[i] - 1L, grid$K)]
      grid$bic_lower_than_previous[i] <- is.finite(previous) && grid$BIC[i] < previous
    }
  }
  grid$entropy_pass <- grid$entropy >= 0.70
  grid$min_APPA_pass <- grid$min_APPA >= 0.80
  grid$min_class_size_pass <- grid$min_class_proportion >= 0.05
  # Clinical interpretability is deliberately left for investigator review;
  # it cannot be decided from a numerical threshold.
  grid$clinical_interpretability_review <- NA
  grid$technical_criteria_pass <- with(
    grid,
    (is.na(bic_lower_than_previous) | bic_lower_than_previous) &
      entropy_pass & min_APPA_pass & min_class_size_pass
  )
  write.csv(grid, file.path(set_dir, paste0("gbmt_model_grid_", set_name, ".csv")), row.names = FALSE)
  all_grid[[set_name]] <- grid
  all_set_counts[[set_name]] <- data.frame(
    set = set_name,
    n_stays = data_obj$n_units,
    n_successful_models = nrow(grid),
    minimum_technical_candidate_K = if (any(grid$technical_criteria_pass)) min(grid$K[grid$technical_criteria_pass]) else NA_integer_,
    stringsAsFactors = FALSE
  )
}

if (length(all_grid) > 0L) {
  combined_grid <- rbind_fill(all_grid)
  rownames(combined_grid) <- NULL
  write.csv(combined_grid, file.path(output_dir, "gbmt_model_grid_all_sets_v1.csv"), row.names = FALSE)
}
if (length(all_set_counts) > 0L) {
  write.csv(do.call(rbind, all_set_counts), file.path(output_dir, "gbmt_set_counts_v1.csv"), row.names = FALSE)
}
if (length(all_failures) > 0L) {
  write.csv(do.call(rbind, all_failures), file.path(output_dir, "gbmt_fit_failures_v1.csv"), row.names = FALSE)
} else {
  write.csv(data.frame(set = character(), K = integer(), error = character()),
            file.path(output_dir, "gbmt_fit_failures_v1.csv"), row.names = FALSE)
}

manifest <- c(
  "script=fit_adhf_gbmt_grid_v1.R",
  "version=strict_plan_GE3_primary_FULL5_sensitivity",
  paste0("input_dir=", input_dir),
  paste0("output_dir=", normalizePath(output_dir, mustWork = FALSE)),
  paste0("sets=", paste(args$sets, collapse = ",")),
  paste0("K_grid=", args$k_min, ":", args$k_max),
  paste0("nstart=", args$nstart),
  paste0("maxit=", args$maxit),
  paste0("tol=", args$tol),
  paste0("seed=", args$seed),
  "polynomial_degree=2",
  "gbmt_scaling=0 (input already pooled marker-specific z-score)",
  "pruning=FALSE (retain linear and quadratic terms)",
  "missingness=no pre-fit imputation; gbmt likelihood handles missing marker values",
  "primary_set=GE3",
  "sensitivity_set=FULL5"
)
writeLines(manifest, file.path(output_dir, "RUN_METADATA.txt"))

cat("\nGBMT grid fitting complete.\n")
cat("Outputs: ", normalizePath(output_dir, mustWork = FALSE), "\n", sep = "")
