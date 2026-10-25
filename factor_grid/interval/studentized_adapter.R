# Candidate adapter only. No fitting or data loading occurs at source time.
# Requires frozen rw_copy_domain(), fit_conditional(), full_design_sandwich(),
# and delta_contrasts(). All intervals are pointwise and conditional on anchors.
candidate_sha256 <- function(path) {
  path <- normalizePath(path,winslash="/",mustWork=TRUE)
  z <- system2("certutil.exe",c("-hashfile",shQuote(path),"SHA256"),stdout=TRUE,stderr=TRUE)
  z <- trimws(z); z <- z[grepl("^[a-f0-9]{64}$",z)]
  if(length(z)!=1L)stop("SHA256 calculation failed")
  z
}

studentized_copy <- function(d,skeleton,draws=NULL) {
  h <- table(as.character(skeleton$region))
  if(any(h<3L))stop("Studentization requires original H >= 3 in every stratum")
  rr <- rw_copy_domain(d,skeleton,seed=NULL,draws=draws)
  # One PSU for EVERY sampled occurrence, including zero-domain copies.
  sk <- data.frame(hh=as.character(rr$trace$copy),region=as.character(rr$trace$region),stringsAsFactors=FALSE)
  hs <- table(sk$region)
  stopifnot(!anyDuplicated(sk$hh),identical(names(hs),names(h)),all(as.numeric(hs)==as.numeric(h)-1L),
            all(hs>=2L),all(rr$data$hh%in%sk$hh),sum(rr$trace$domain_rows)==nrow(rr$data))
  mapped <- match(rr$data$hh,sk$hh)
  stopifnot(all(as.character(rr$data$region)==sk$region[mapped]),
            all(abs(rr$trace$scale-as.numeric(h[rr$trace$region])/(as.numeric(h[rr$trace$region])-1))<1e-12))
  rr$skeleton <- sk
  rr$audit <- list(original_households=nrow(skeleton),copy_households=nrow(sk),strata=length(h),
    copy_zero_domain=sum(rr$trace$domain_rows==0L),copy_domain_households=sum(rr$trace$domain_rows>0L),
    copy_domain_rows=nrow(rr$data),original_H=as.list(h),copy_H=as.list(hs),
    copy_finite_factors=as.list(hs/(hs-1)),copy_id_unique=TRUE,full_skeleton_retained=TRUE)
  rr
}

studentized_analyze <- function(d,skeleton,structure="M1R") {
  if(!identical(structure,"M1R"))stop("This candidate is frozen to M1R")
  f <- fit_conditional(d,structure)
  ans <- data.frame(item=paste0("i",5:8),estimate=NA_real_,se=NA_real_,fit_ok=isTRUE(f$ok),se_ok=FALSE)
  warnings <- f$warnings;error <- f$error;diag <- NULL
  if(isTRUE(f$ok)) {
    result <- withCallingHandlers(tryCatch({
      # Preserve point estimates even if variance construction fails.
      point <- delta_contrasts(f$fit,matrix(0,length(coef(f$fit)),length(coef(f$fit))),anchors=anchors)
      stopifnot(identical(point$item,ans$item));ans$estimate <- point$estimate
      cs <- full_design_sandwich(f$fit,d,skeleton,finite=TRUE,criterion="objective")
      an <- delta_contrasts(f$fit,cs$vcov,anchors=anchors)
      stopifnot(identical(an$item,ans$item));ans$se <- an$se
      ans$se_ok <- is.finite(an$se)&an$se>0&is.finite(an$estimate)
      diag <- cs$diagnostics
      if(!all(ans$se_ok))error <- "One or more target SEs are invalid"
      TRUE
    },error=function(e)e),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
    if(inherits(result,"error"))error <- conditionMessage(result)
  }
  list(targets=ans,fit_ok=isTRUE(f$ok),analysis_ok=all(ans$se_ok),error=error,warnings=warnings,diagnostics=diag)
}

studentized_intervals <- function(original,replicates,B=399L,alpha=.05,type=6L,df) {
  stopifnot(B==399L,identical(as.integer(type),6L),alpha==.05,
            identical(original$item,paste0("i",5:8)),is.finite(df),df>0)
  out <- vector("list",4L)
  for(j in seq_len(4L)) {
    o <- original[j,,drop=FALSE];r <- replicates[replicates$item==o$item,,drop=FALSE]
    if(nrow(r)!=B||!identical(sort(as.integer(r$replicate)),seq_len(B)))stop("Fixed replicate slots missing/duplicated")
    r <- r[order(r$replicate),,drop=FALSE]
    point_valid <- r$fit_ok&is.finite(r$estimate)
    t_valid <- point_valid&r$se_ok&is.finite(r$se)&r$se>0
    origin_point_valid <- isTRUE(o$fit_ok)&&is.finite(o$estimate)
    origin_valid <- origin_point_valid&&isTRUE(o$se_ok)&&is.finite(o$se)&&o$se>0
    v <- data.frame(item=o$item,estimate=o$estimate,analytic_se=o$se,B_requested=B,
      B_point_valid=sum(point_valid),B_studentized_valid=sum(t_valid),analytic_defined=origin_valid,
      studentized_defined=origin_valid&&all(t_valid),basic_defined=origin_point_valid&&all(point_valid),
      percentile_defined=origin_point_valid&&all(point_valid),analytic_lower=NA_real_,analytic_upper=NA_real_,
      studentized_lower=NA_real_,studentized_upper=NA_real_,t_q025=NA_real_,t_q975=NA_real_,
      basic_lower=NA_real_,basic_upper=NA_real_,percentile_lower=NA_real_,percentile_upper=NA_real_,
      candidate="bootstrap-t",descriptive_only_comparators="analytic/basic/percentile",quantile_type=type,
      pointwise=TRUE,familywise_adjusted=FALSE,stringsAsFactors=FALSE)
    if(origin_valid){cr<-qt(1-alpha/2,df);v$analytic_lower<-o$estimate-cr*o$se;v$analytic_upper<-o$estimate+cr*o$se}
    if(v$studentized_defined) {
      tstar <- (r$estimate-o$estimate)/r$se
      if(any(!is.finite(tstar)))stop("Invalid studentized statistic despite validity gate")
      qq <- quantile(tstar,c(alpha/2,1-alpha/2),type=type,names=FALSE,na.rm=FALSE)
      v$t_q025<-qq[1];v$t_q975<-qq[2]
      v$studentized_lower<-o$estimate-qq[2]*o$se;v$studentized_upper<-o$estimate-qq[1]*o$se
    }
    if(origin_point_valid&&all(point_valid)) {
      qq<-quantile(r$estimate,c(alpha/2,1-alpha/2),type=type,names=FALSE,na.rm=FALSE)
      v$percentile_lower<-qq[1];v$percentile_upper<-qq[2]
      if(v$basic_defined){v$basic_lower<-2*o$estimate-qq[2];v$basic_upper<-2*o$estimate-qq[1]}
    }
    v$studentized_outside_parameter_support <- if(v$studentized_defined) v$studentized_lower< -1||v$studentized_upper>1 else NA
    out[[j]]<-v
  }
  do.call(rbind,out)
}
