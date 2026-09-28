#!/usr/bin/env Rscript

# Downstream analysis for the strict-plan ADHF trajectory cohort.
#
# This script does not refit gbmt. It reads the saved modal assignments and
# posterior probabilities, joins them to the landmark outcome table, and
# produces class summaries, trajectory summaries, Kaplan-Meier curves, and
# Cox models. It can analyse K=3, K=4, and K=5 from the existing grid; this is
# useful when K=4 is the primary model and K +/- 1 are supplementary analyses.

options(stringsAsFactors = FALSE)

parse_cli <- function(args) {
  out <- list(
    model_input_dir = NULL,
    trajectory_dir = NULL,
    gbmt_dir = NULL,
    cohort = NULL,
    output_dir = NULL,
    sets = c("GE3", "FULL5"),
    ks = c(3L, 4L, 5L),
    primary_set = "GE3",
    primary_k = 4L,
    sensitivity_set = "FULL5",
    covariates = "",
    seed = 20260915L
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
      out[[key]] <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
    } else if (key == "ks") {
      out[[key]] <- as.integer(trimws(strsplit(value, ",", fixed = TRUE)[[1L]]))
    } else if (key %in% c("primary_k", "seed")) {
      out[[key]] <- as.integer(value)
    } else {
      out[[key]] <- value
    }
  }
  out
}

args <- parse_cli(commandArgs(trailingOnly = TRUE))
if (is.null(args$model_input_dir) || is.null(args$trajectory_dir) ||
    is.null(args$gbmt_dir) || is.null(args$output_dir)) {
  stop(
    paste(
      "Usage:",
      "Rscript analyze_adhf_gbmt_outcomes_v1.R",
      "--model_input_dir <trajectory_model_inputs_v1_1458>",
      "--trajectory_dir <trajectory_prepare_v2_1458>",
      "--gbmt_dir <gbmt_results_v1_1458>",
      "--output_dir <gbmt_outcomes_v1_1458>",
      "[--cohort <adhf_icu_cohort_v4.csv>]",
      "[--sets GE3,FULL5] [--ks 3,4,5]",
      "[--primary_set GE3] [--primary_k 4]",
      "[--covariates age_at_admission,gender,sofa,sapsii]"
    ),
    call. = FALSE
  )
}

if (!requireNamespace("survival", quietly = TRUE)) {
  stop("The survival package is required. Install it with install.packages('survival').", call. = FALSE)
}

model_input_dir <- normalizePath(path.expand(args$model_input_dir), mustWork = TRUE)
trajectory_dir <- normalizePath(path.expand(args$trajectory_dir), mustWork = TRUE)
gbmt_dir <- normalizePath(path.expand(args$gbmt_dir), mustWork = TRUE)
output_dir <- path.expand(args$output_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (length(args$sets) == 0L || any(!args$sets %in% c("GE2", "GE3", "FULL5"))) {
  stop("--sets must contain only GE2, GE3, and/or FULL5.", call. = FALSE)
}
if (length(args$ks) == 0L || any(is.na(args$ks)) || any(args$ks < 2L)) {
  stop("--ks must contain integers >= 2.", call. = FALSE)
}
if (!(args$primary_set %in% args$sets) || !(args$primary_k %in% args$ks) ||
    !(args$sensitivity_set %in% args$sets)) {
  stop("--primary_set, --sensitivity_set, and --primary_k must be included in --sets and --ks.", call. = FALSE)
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

read_csv_checked <- function(path, label = basename(path)) {
  if (!file.exists(path)) stop("File not found: ", path, call. = FALSE)
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
           na.strings = c("", "NA", "NaN"))
}

id_key <- function(data) {
  paste(data$subject_id, data$hadm_id, data$stay_id, sep = "|")
}

safe_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (sum(is.finite(x)) == 0L) return(NA_real_)
  mean(x[is.finite(x)])
}

safe_sd <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(NA_real_)
  stats::sd(x)
}

safe_quantile <- function(x, p) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  as.numeric(stats::quantile(x, probs = p, names = FALSE, type = 7))
}

safe_rate <- function(event, denominator) {
  event <- suppressWarnings(as.numeric(event))
  denominator <- suppressWarnings(as.numeric(denominator))
  if (!is.finite(denominator) || denominator <= 0) return(NA_real_)
  sum(event == 1, na.rm = TRUE) / denominator
}

read_outcomes <- function() {
  path <- file.path(trajectory_dir, "landmark_outcomes_v1.csv")
  out <- read_csv_checked(path, "landmark_outcomes_v1.csv")
  stop_missing(
    out,
    c(id_cols, "day5_landmark_eligible", "mortality_90d",
      "in_hospital_mortality", "time_to_death_or_90d_days"),
    basename(path)
  )
  for (nm in c("day5_landmark_eligible", "mortality_90d",
               "in_hospital_mortality", "time_to_death_or_90d_days")) {
    out[[nm]] <- suppressWarnings(as.numeric(out[[nm]]))
  }
  out$key__ <- id_key(out)
  if (anyDuplicated(out$key__)) stop("Duplicate stay IDs in landmark outcomes.", call. = FALSE)
  out
}

read_cohort <- function() {
  if (is.null(args$cohort) || identical(args$cohort, "")) return(NULL)
  path <- normalizePath(path.expand(args$cohort), mustWork = TRUE)
  cohort <- read_csv_checked(path)
  stop_missing(cohort, id_cols, basename(path))
  cohort$key__ <- id_key(cohort)
  if (anyDuplicated(cohort$key__)) {
    cohort <- cohort[!duplicated(cohort$key__), , drop = FALSE]
  }
  cohort
}

read_assignments <- function(set_name, k) {
  path <- file.path(
    gbmt_dir, set_name,
    paste0("gbmt_", set_name, "_K", k, "_assignments.csv")
  )
  a <- read_csv_checked(path)
  stop_missing(a, c(id_cols, "assigned_class", "max_posterior"), basename(path))
  for (nm in id_cols) a[[nm]] <- suppressWarnings(as.numeric(a[[nm]]))
  a$assigned_class <- suppressWarnings(as.integer(a$assigned_class))
  a$max_posterior <- suppressWarnings(as.numeric(a$max_posterior))
  a$key__ <- id_key(a)
  if (anyDuplicated(a$key__)) stop("Duplicate stay IDs in ", path, ".", call. = FALSE)
  posterior_cols <- grep("^posterior_class_", names(a), value = TRUE)
  for (nm in posterior_cols) a[[nm]] <- suppressWarnings(as.numeric(a[[nm]]))
  a
}

read_daily <- function(set_name) {
  daily_path <- file.path(
    model_input_dir,
    paste0("trajectory_daily_scaled_", set_name, "_v1.csv")
  )
  if (file.exists(daily_path)) {
    d <- read_csv_checked(daily_path)
    stop_missing(d, c(id_cols, "day_num", markers), basename(daily_path))
    return(d)
  }

  long_path <- file.path(
    model_input_dir,
    paste0("trajectory_long_scaled_", set_name, "_v1.csv")
  )
  long <- read_csv_checked(long_path)
  stop_missing(long, c(id_cols, "day_num", "marker", "value_z"), basename(long_path))
  long$day_num <- suppressWarnings(as.numeric(long$day_num))
  long$value_z <- suppressWarnings(as.numeric(long$value_z))
  long$marker <- as.character(long$marker)
  if (any(!long$marker %in% markers)) {
    stop("Unexpected marker in ", basename(long_path), ".", call. = FALSE)
  }
  wide <- reshape(
    long[, c(id_cols, "day_num", "marker", "value_z")],
    idvar = c(id_cols, "day_num"),
    timevar = "marker",
    direction = "wide"
  )
  names(wide) <- sub("^value_z\\.", "", names(wide))
  stop_missing(wide, c(id_cols, "day_num", markers), "reshaped trajectory data")
  wide
}

merge_assignment_data <- function(assignments, daily, outcomes, cohort = NULL) {
  daily$key__ <- id_key(daily)
  if (anyDuplicated(daily[, c("key__", "day_num")])) {
    stop("Duplicate stay-day rows in trajectory input.", call. = FALSE)
  }
  daily <- merge(
    daily,
    assignments[, c("key__", "assigned_class", "max_posterior"), drop = FALSE],
    by = "key__", all = FALSE, sort = FALSE
  )
  daily <- merge(
    daily,
    outcomes[, c("key__", "day5_landmark_eligible", "mortality_90d",
                 "in_hospital_mortality", "time_to_death_or_90d_days"), drop = FALSE],
    by = "key__", all.x = TRUE, sort = FALSE
  )
  if (!is.null(cohort)) {
    extras <- setdiff(names(cohort), c(id_cols, "key__"))
    extras <- setdiff(extras, names(daily))
    if (length(extras) > 0L) {
      daily <- merge(
        daily,
        cohort[, c("key__", extras), drop = FALSE],
        by = "key__", all.x = TRUE, sort = FALSE
      )
    }
  }
  daily
}

class_summary <- function(assignments, outcomes, set_name, k) {
  merged <- merge(
    assignments,
    outcomes[, c("key__", "day5_landmark_eligible", "mortality_90d",
                 "in_hospital_mortality"), drop = FALSE],
    by = "key__", all.x = TRUE, sort = FALSE
  )
  rows <- lapply(seq_len(k), function(g) {
    z <- merged[merged$assigned_class == g, , drop = FALSE]
    eligible <- z$day5_landmark_eligible == 1
    data.frame(
      set = set_name,
      K = k,
      assigned_class = g,
      n_stays = nrow(z),
      proportion = nrow(z) / nrow(merged),
      mean_max_posterior = safe_mean(z$max_posterior),
      n_day5_landmark_eligible = sum(eligible, na.rm = TRUE),
      n_90d_deaths_all = sum(z$mortality_90d == 1, na.rm = TRUE),
      n_90d_deaths_day5_eligible = sum(z$mortality_90d[eligible] == 1, na.rm = TRUE),
      n_in_hospital_deaths_all = sum(z$in_hospital_mortality == 1, na.rm = TRUE),
      n_in_hospital_deaths_day5_eligible = sum(z$in_hospital_mortality[eligible] == 1, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

posterior_summary <- function(assignments, set_name, k) {
  posterior_cols <- grep("^posterior_class_", names(assignments), value = TRUE)
  if (length(posterior_cols) == 0L) return(data.frame())
  rows <- lapply(posterior_cols, function(nm) {
    x <- assignments[[nm]]
    data.frame(
      set = set_name,
      K = k,
      posterior_column = nm,
      mean = safe_mean(x),
      sd = safe_sd(x),
      q05 = safe_quantile(x, 0.05),
      median = safe_quantile(x, 0.50),
      q95 = safe_quantile(x, 0.95),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

trajectory_summary <- function(daily, set_name, k) {
  classes <- sort(unique(daily$assigned_class))
  days <- sort(unique(suppressWarnings(as.numeric(daily$day_num))))
  rows <- list()
  idx <- 0L
  for (g in classes) {
    for (day in days) {
      z <- daily[daily$assigned_class == g & daily$day_num == day, , drop = FALSE]
      row <- data.frame(
        set = set_name,
        K = k,
        assigned_class = g,
        day_num = day,
        n_stays = nrow(z),
        stringsAsFactors = FALSE
      )
      for (m in markers) {
        v <- suppressWarnings(as.numeric(z[[m]]))
        row[[paste0(m, "_mean")]] <- safe_mean(v)
        row[[paste0(m, "_sd")]] <- safe_sd(v)
        row[[paste0(m, "_n_observed")]] <- sum(is.finite(v))
      }
      idx <- idx + 1L
      rows[[idx]] <- row
    }
  }
  do.call(rbind, rows)
}

outcome_summary <- function(merged, set_name, k) {
  classes <- sort(unique(merged$assigned_class))
  rows <- lapply(classes, function(g) {
    z <- merged[merged$assigned_class == g, , drop = FALSE]
    eligible <- z$day5_landmark_eligible == 1
    d90 <- z$mortality_90d[eligible]
    dh <- z$in_hospital_mortality[eligible]
    data.frame(
      set = set_name,
      K = k,
      assigned_class = g,
      n_stays = nrow(z),
      n_day5_landmark_eligible = sum(eligible, na.rm = TRUE),
      n_90d_deaths_day5_eligible = sum(d90 == 1, na.rm = TRUE),
      mortality_90d_day5_eligible = safe_rate(d90, sum(!is.na(d90))),
      n_in_hospital_deaths_day5_eligible = sum(dh == 1, na.rm = TRUE),
      in_hospital_mortality_day5_eligible = safe_rate(dh, sum(!is.na(dh))),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

cox_table <- function(fit, model_label, set_name, k, n_used, n_events) {
  if (is.null(fit)) {
    return(data.frame(
      model = model_label, set = set_name, K = k, n_used = n_used,
      n_events = n_events, term = character(), log_hazard = numeric(),
      hazard_ratio = numeric(), conf_low = numeric(), conf_high = numeric(),
      se = numeric(), z = numeric(), p = numeric(), stringsAsFactors = FALSE
    ))
  }
  s <- summary(fit)
  co <- as.data.frame(s$coefficients)
  ci <- as.data.frame(s$conf.int)
  data.frame(
    model = model_label,
    set = set_name,
    K = k,
    n_used = n_used,
    n_events = n_events,
    term = rownames(co),
    log_hazard = as.numeric(co[, "coef"]),
    hazard_ratio = as.numeric(ci[, "exp(coef)"]),
    conf_low = as.numeric(ci[, "lower .95"]),
    conf_high = as.numeric(ci[, "upper .95"]),
    se = as.numeric(co[, "se(coef)"]),
    z = as.numeric(co[, "z"]),
    p = as.numeric(co[, "Pr(>|z|)"]),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

fit_cox_models <- function(assignments, outcomes, cohort, set_name, k, is_primary = FALSE) {
  dat <- merge(
    assignments,
    outcomes,
    by = "key__", all = FALSE, sort = FALSE
  )
  if (!is.null(cohort)) {
    extras <- setdiff(names(cohort), c(id_cols, "key__"))
    extras <- setdiff(extras, names(dat))
    if (length(extras) > 0L) {
      dat <- merge(dat, cohort[, c("key__", extras), drop = FALSE],
                   by = "key__", all.x = TRUE, sort = FALSE)
    }
  }
  eligible <- dat[
    dat$day5_landmark_eligible == 1 &
      is.finite(dat$time_to_death_or_90d_days) &
      !is.na(dat$mortality_90d),
    , drop = FALSE
  ]
  if (nrow(eligible) == 0L || sum(eligible$mortality_90d == 1, na.rm = TRUE) == 0L) {
    warning("No usable day-5 landmark events for ", set_name, " K=", k, ".")
    return(list(
      data = dat,
      eligible = eligible,
      unadjusted = data.frame(),
      adjusted = data.frame(),
      ph = data.frame()
    ))
  }
  eligible$assigned_class <- factor(eligible$assigned_class)
  eligible$assigned_class <- stats::relevel(eligible$assigned_class,
                                             ref = as.character(min(as.integer(as.character(eligible$assigned_class)))))
  form_unadj <- stats::as.formula(
    "survival::Surv(time_to_death_or_90d_days, mortality_90d) ~ assigned_class"
  )
  fit_unadj <- tryCatch(
    survival::coxph(form_unadj, data = eligible, ties = "efron", x = TRUE, y = TRUE),
    error = function(e) {
      warning("Unadjusted Cox failed for ", set_name, " K=", k, ": ", conditionMessage(e))
      NULL
    }
  )
  unadj <- cox_table(
    fit_unadj, "unadjusted", set_name, k, nrow(eligible),
    sum(eligible$mortality_90d == 1, na.rm = TRUE)
  )

  adjusted <- data.frame()
  fit_adj <- NULL
  requested_covars <- character()
  if (!is.null(args$covariates) && nzchar(trimws(args$covariates))) {
    requested_covars <- trimws(strsplit(args$covariates, ",", fixed = TRUE)[[1L]])
    requested_covars <- requested_covars[nzchar(requested_covars)]
    present <- intersect(requested_covars, names(eligible))
    absent <- setdiff(requested_covars, present)
    if (length(absent) > 0L) {
      writeLines(
        paste0("Not found in cohort: ", paste(absent, collapse = ", ")),
        file.path(output_dir, paste0("missing_covariates_", set_name, "_K", k, ".txt"))
      )
    }
    if (length(present) > 0L) {
      for (nm in present) {
        if (is.character(eligible[[nm]]) || is.logical(eligible[[nm]])) {
          eligible[[nm]] <- factor(eligible[[nm]])
        } else {
          eligible[[nm]] <- suppressWarnings(as.numeric(eligible[[nm]]))
        }
      }
      keep <- c("time_to_death_or_90d_days", "mortality_90d", "assigned_class", present)
      complete <- eligible[stats::complete.cases(eligible[, keep, drop = FALSE]), , drop = FALSE]
      if (nrow(complete) > 0L && sum(complete$mortality_90d == 1, na.rm = TRUE) > 0L) {
        terms <- c("assigned_class", paste0("`", present, "`"))
        form_adj <- stats::as.formula(
          paste0("survival::Surv(time_to_death_or_90d_days, mortality_90d) ~ ",
                 paste(terms, collapse = " + "))
        )
        fit_adj <- tryCatch(
          survival::coxph(form_adj, data = complete, ties = "efron", x = TRUE, y = TRUE),
          error = function(e) {
            warning("Adjusted Cox failed for ", set_name, " K=", k, ": ", conditionMessage(e))
            NULL
          }
        )
        adjusted <- cox_table(
          fit_adj, "adjusted", set_name, k, nrow(complete),
          sum(complete$mortality_90d == 1, na.rm = TRUE)
        )
      }
    }
  }

  ph <- data.frame()
  if (!is.null(fit_unadj)) {
    ph_obj <- tryCatch(survival::cox.zph(fit_unadj), error = function(e) NULL)
    if (!is.null(ph_obj)) {
      ph <- data.frame(
        model = "unadjusted",
        set = set_name,
        K = k,
        term = rownames(ph_obj$table),
        chisq = as.numeric(ph_obj$table[, "chisq"]),
        df = as.numeric(ph_obj$table[, "df"]),
        p = as.numeric(ph_obj$table[, "p"]),
        row.names = NULL,
        stringsAsFactors = FALSE
      )
    }
  }

  if (is_primary && !is.null(fit_unadj)) {
    km <- tryCatch(survival::survfit(form_unadj, data = eligible), error = function(e) NULL)
    if (!is.null(km)) {
      pdf(file.path(output_dir, paste0("KM_", set_name, "_K", k, "_day5_landmark_90d.pdf")),
          width = 8, height = 6, useDingbats = FALSE)
      plot(km, col = seq_len(k), lwd = 2, mark.time = TRUE,
           xlab = "Days after day-5 landmark", ylab = "Survival probability",
           main = paste0("90-day survival: ", set_name, " K=", k))
      legend("bottomleft", legend = paste0("Class ", seq_len(k)),
             col = seq_len(k), lwd = 2, bty = "n")
      dev.off()
      km_sum <- summary(km, times = c(30, 60, 90))
      if (length(km_sum$time) > 0L) {
        write.csv(
          data.frame(
            set = set_name, K = k,
            time_days = km_sum$time,
            n_risk = km_sum$n.risk,
            n_event = km_sum$n.event,
            survival = km_sum$surv,
            std_err = km_sum$std.err,
            strata = if (is.null(km_sum$strata)) NA_character_ else as.character(km_sum$strata),
            stringsAsFactors = FALSE
          ),
          file.path(output_dir, paste0("KM_points_", set_name, "_K", k, ".csv")),
          row.names = FALSE
        )
      }
      logrank <- tryCatch(survival::survdiff(form_unadj, data = eligible), error = function(e) NULL)
      if (!is.null(logrank)) {
        write.csv(
          data.frame(
            set = set_name,
            K = k,
            chisq = unname(logrank$chisq),
            df = length(logrank$n) - 1L,
            p_value = stats::pchisq(logrank$chisq, df = length(logrank$n) - 1L, lower.tail = FALSE),
            stringsAsFactors = FALSE
          ),
          file.path(output_dir, paste0("logrank_", set_name, "_K", k, ".csv")),
          row.names = FALSE
        )
      }
    }
  }

  list(data = dat, eligible = eligible, unadjusted = unadj,
       adjusted = adjusted, ph = ph)
}

outcomes <- read_outcomes()
cohort <- read_cohort()

all_class <- list()
all_post <- list()
all_traj <- list()
all_outcome <- list()
all_unadj <- list()
all_adj <- list()
all_ph <- list()
assignment_cache <- list()

for (set_name in args$sets) {
  for (k in args$ks) {
    message("Reading ", set_name, " K=", k, "...")
    assignment_path <- file.path(
      gbmt_dir, set_name,
      paste0("gbmt_", set_name, "_K", k, "_assignments.csv")
    )
    if (!file.exists(assignment_path)) {
      warning("Skipping missing assignment file: ", assignment_path)
      next
    }
    assignments <- read_assignments(set_name, k)
    daily <- read_daily(set_name)
    merged_daily <- merge_assignment_data(assignments, daily, outcomes, cohort)
    merged_outcome <- merge(
      assignments,
      outcomes,
      by = "key__", all = FALSE, sort = FALSE
    )
    if (!is.null(cohort)) {
      extras <- setdiff(names(cohort), c(id_cols, "key__"))
      extras <- setdiff(extras, names(merged_outcome))
      if (length(extras) > 0L) {
        merged_outcome <- merge(
          merged_outcome,
          cohort[, c("key__", extras), drop = FALSE],
          by = "key__", all.x = TRUE, sort = FALSE
        )
      }
    }

    cls <- class_summary(assignments, outcomes, set_name, k)
    post <- posterior_summary(assignments, set_name, k)
    traj <- trajectory_summary(merged_daily, set_name, k)
    outc <- outcome_summary(merged_outcome, set_name, k)
    cox <- fit_cox_models(
      assignments, outcomes, cohort, set_name, k,
      is_primary = identical(set_name, args$primary_set) && identical(k, args$primary_k)
    )

    prefix <- paste0(set_name, "_K", k)
    write.csv(cls, file.path(output_dir, paste0("class_summary_", prefix, ".csv")), row.names = FALSE)
    write.csv(post, file.path(output_dir, paste0("posterior_summary_", prefix, ".csv")), row.names = FALSE)
    write.csv(traj, file.path(output_dir, paste0("trajectory_class_summary_", prefix, ".csv")), row.names = FALSE)
    write.csv(outc, file.path(output_dir, paste0("outcome_summary_", prefix, ".csv")), row.names = FALSE)
    write.csv(cox$unadjusted, file.path(output_dir, paste0("cox_unadjusted_", prefix, ".csv")), row.names = FALSE)
    write.csv(cox$adjusted, file.path(output_dir, paste0("cox_adjusted_", prefix, ".csv")), row.names = FALSE)
    write.csv(cox$ph, file.path(output_dir, paste0("cox_ph_test_", prefix, ".csv")), row.names = FALSE)

    all_class[[length(all_class) + 1L]] <- cls
    all_post[[length(all_post) + 1L]] <- post
    all_traj[[length(all_traj) + 1L]] <- traj
    all_outcome[[length(all_outcome) + 1L]] <- outc
    all_unadj[[length(all_unadj) + 1L]] <- cox$unadjusted
    all_adj[[length(all_adj) + 1L]] <- cox$adjusted
    all_ph[[length(all_ph) + 1L]] <- cox$ph
    assignment_cache[[paste0(set_name, "_K", k)]] <- assignments
  }
}

bind_nonempty <- function(x) {
  x <- x[vapply(x, nrow, integer(1)) > 0L]
  if (length(x) == 0L) return(data.frame())
  do.call(rbind, x)
}

write.csv(bind_nonempty(all_class), file.path(output_dir, "class_summary_all_models_v1.csv"), row.names = FALSE)
write.csv(bind_nonempty(all_post), file.path(output_dir, "posterior_summary_all_models_v1.csv"), row.names = FALSE)
write.csv(bind_nonempty(all_traj), file.path(output_dir, "trajectory_class_summary_all_models_v1.csv"), row.names = FALSE)
write.csv(bind_nonempty(all_outcome), file.path(output_dir, "outcome_summary_all_models_v1.csv"), row.names = FALSE)
write.csv(bind_nonempty(all_unadj), file.path(output_dir, "cox_unadjusted_all_models_v1.csv"), row.names = FALSE)
write.csv(bind_nonempty(all_adj), file.path(output_dir, "cox_adjusted_all_models_v1.csv"), row.names = FALSE)
write.csv(bind_nonempty(all_ph), file.path(output_dir, "cox_ph_test_all_models_v1.csv"), row.names = FALSE)

# Crosswalk between GE3 and FULL5 for the selected K. The numeric labels are
# model-specific; the table is descriptive and does not assume label identity.
cross_key_ge3 <- paste0(args$primary_set, "_K", args$primary_k)
cross_key_full5 <- paste0(args$sensitivity_set, "_K", args$primary_k)
if (cross_key_ge3 %in% names(assignment_cache) && cross_key_full5 %in% names(assignment_cache)) {
  a <- assignment_cache[[cross_key_ge3]][, c("key__", "assigned_class"), drop = FALSE]
  b <- assignment_cache[[cross_key_full5]][, c("key__", "assigned_class"), drop = FALSE]
  names(a)[2] <- "class_primary"
  names(b)[2] <- "class_sensitivity"
  cross <- merge(a, b, by = "key__", all = FALSE)
  tab <- as.data.frame(table(cross$class_primary, cross$class_sensitivity), stringsAsFactors = FALSE)
  names(tab) <- c("primary_class", "sensitivity_class", "n_stays")
  tab$primary_class <- as.integer(as.character(tab$primary_class))
  tab$sensitivity_class <- as.integer(as.character(tab$sensitivity_class))
  row_totals <- tapply(tab$n_stays, tab$primary_class, sum)
  col_totals <- tapply(tab$n_stays, tab$sensitivity_class, sum)
  tab$within_primary_proportion <- tab$n_stays / row_totals[as.character(tab$primary_class)]
  tab$within_sensitivity_proportion <- tab$n_stays / col_totals[as.character(tab$sensitivity_class)]
  tab$overlap_n <- nrow(cross)
  write.csv(tab, file.path(output_dir, "crosswalk_primary_vs_FULL5_v1.csv"), row.names = FALSE)
}

metadata <- c(
  "script=analyze_adhf_gbmt_outcomes_v1.R",
  "version=strict_plan_downstream_outcomes",
  paste0("model_input_dir=", model_input_dir),
  paste0("trajectory_dir=", trajectory_dir),
  paste0("gbmt_dir=", gbmt_dir),
  paste0("output_dir=", normalizePath(output_dir, mustWork = FALSE)),
  paste0("sets=", paste(args$sets, collapse = ",")),
  paste0("ks=", paste(args$ks, collapse = ",")),
  paste0("primary_set=", args$primary_set),
  paste0("primary_k=", args$primary_k),
  paste0("sensitivity_set=", args$sensitivity_set),
  paste0("covariates=", args$covariates),
  "outcome=90-day mortality from the day-5 landmark among landmark-eligible stays",
  "assignment=modal gbmt class with posterior probabilities retained",
  "survival=unadjusted Cox; adjusted Cox is run only for covariates supplied and present in the cohort",
  "no_gbmt_refit=true"
)
writeLines(metadata, file.path(output_dir, "RUN_METADATA.txt"))

cat("\nDownstream ADHF trajectory analysis complete.\n")
cat("Outputs: ", normalizePath(output_dir, mustWork = FALSE), "\n", sep = "")
