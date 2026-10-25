# Synthetic calibration support for complete binary ordinal DWLS models only.
# First-order threshold/tetrachoric moment influence functions, grouped by
# household and centered within observed strata. Model derivative uses DWLS's
# expected-information Gauss-Newton form. No claim of complex-survey validity
# beyond independent PSUs within strata and the supplied fixed case weights.

cluster_sandwich <- function(fit,d,group="O",weight="w",cluster="hh",stratum="region",finite=TRUE,debug=FALSE) {
  stopifnot(inherits(fit,"lavaan"),all(c(group,weight,cluster,stratum)%in%names(d)))
  opt <- lavaan::lavInspect(fit,"options")
  if(opt$estimator!="DWLS")stop("Only DWLS is implemented")
  if(opt$sampling.weights.normalization!="group")stop("Require group-normalized sampling weights")
  if(anyNA(d[,c(group,weight,cluster,stratum)]))stop("Missing design information")
  if(any(!is.finite(d[[weight]]))||any(d[[weight]]<=0))stop("Weights must be positive finite")
  gl <- lavaan::lavInspect(fit,"group.label")
  obs <- lavaan::lavInspect(fit,"wls.obs")
  delta <- lavaan::lavTech(fit,"delta")
  WV <- lavaan::lavInspect(fit,"wls.v")
  ng <- as.numeric(lavaan::lavInspect(fit,"nobs"));N <- sum(ng)
  # Group-normalized weights can leave nobs at integer +/- 2e-13 in lavaan.
  # This numerical tolerance is far below one record and does not permit
  # a changed analysis sample; no rounding enters the covariance formula.
  count_tolerance <- 1e-7
  if(any(!is.finite(ng))||abs(N-nrow(d))>count_tolerance)stop("Complete-case input must exactly match fitted records")
  if(length(gl)!=2L)stop("Only two groups implemented")
  dims <- lengths(obs);offset <- c(0L,cumsum(dims))
  IF <- matrix(0,nrow(d),sum(dims))
  moment_center <- list(); margin_error <- 0; min_cell <- Inf
  for(g in seq_along(gl)) {
    ix <- which(as.character(d[[group]])==gl[g]); dg <- d[ix,,drop=FALSE]
    if(abs(length(ix)-ng[g])>count_tolerance)stop("Fitted group counts do not match")
    nm <- names(obs[[g]]);s <- as.numeric(obs[[g]])
    th <- which(grepl("|",nm,fixed=TRUE));co <- which(grepl("~~",nm,fixed=TRUE))
    vn <- sub("\\|t1$","",nm[th])
    if(length(th)!=8L || length(co)!=28L || any(!vn%in%names(dg)))stop("Require eight binary variables")
    Y <- as.matrix(dg[,vn,drop=FALSE]);if(anyNA(Y)||any(!Y%in%c(0,1)))stop("Complete 0/1 items only")
    w <- dg[[weight]];w <- w/sum(w)
    p0 <- colSums((Y==0)*w);tau <- s[th]
    margin_error <- max(margin_error,max(abs(pnorm(tau)-p0)))
    if(margin_error>1e-5)stop("Estimated thresholds are not empirical weighted marginal quantiles")
    It <- sweep(Y==0,2,p0,"-")
    It <- sweep(It,2,dnorm(tau),"/")
    raw <- matrix(0,nrow(Y),length(s));raw[,th] <- It
    for(k in co) {
      pair <- strsplit(nm[k],"~~",fixed=TRUE)[[1]];a <- match(pair[1],vn);b <- match(pair[2],vn)
      if(anyNA(c(a,b)))stop("Unknown tetrachoric pair")
      r <- s[k];if(!is.finite(r)||abs(r)>=.999)stop("Tetrachoric boundary")
      ta<-tau[a];tb<-tau[b];sr<-sqrt(1-r*r)
      deriv_a<-dnorm(ta)*pnorm((tb-r*ta)/sr)
      deriv_b<-dnorm(tb)*pnorm((ta-r*tb)/sr)
      deriv_r<-exp(-(ta*ta-2*r*ta*tb+tb*tb)/(2*(1-r*r)))/(2*pi*sr)
      z00<-(Y[,a]==0)&(Y[,b]==0);p00<-sum(w*z00)
      cells<-c(p00,p0[a]-p00,p0[b]-p00,1-p0[a]-p0[b]+p00)
      min_cell<-min(min_cell,cells)
      if(any(cells<=1e-8))stop("Empty bivariate cell: continuity corrections outside scope")
      raw[,k]<-(as.numeric(z00)-p00-deriv_a*It[,a]-deriv_b*It[,b])/deriv_r
    }
    IF[ix,offset[g]+seq_len(dims[g])]<-raw*w
    moment_center[[g]]<-colSums(raw*w)
  }
  # Every free parameter appears exactly once, except an optional repeated
  # appearance of the same free id in lavaan's parameter table.
  pt<-lavaan::parTable(fit);npar<-length(lavaan::coef(fit))
  freept<-pt[pt$free>0,,drop=FALSE];freept<-freept[match(seq_len(npar),freept$free),,drop=FALSE]
  if(anyNA(freept$free)||any(freept$free!=seq_len(npar)))stop("Unexpected free parameter table")
  key<-ifelse(nchar(freept$label)>0,freept$label,paste0("FREE_",freept$free))
  unique_key<-unique(key);E<-outer(key,unique_key,"==")*1
  constraints<-pt[pt$op%in%c("==","<",">",":="),,drop=FALSE]
  if(any(constraints$op!="=="))stop("Non-simple constraints not supported")
  if(nrow(constraints)!=(npar-ncol(E)))stop("Constraint count does not match repeated equality labels")
  # Check the actual constraint endpoints, not merely their count.
  if(nrow(constraints)>0)for(k in seq_len(nrow(constraints))) {
    left<-match(constraints$lhs[k],freept$plabel);right<-match(constraints$rhs[k],freept$plabel)
    if(anyNA(c(left,right))||key[left]!=key[right])stop("Constraint is not a repeated-label equality")
  }
  nuniq<-ncol(E);bread<-matrix(0,nuniq,nuniq);DWs<-list()
  for(g in seq_along(gl)) {
    D<-delta[[g]]%*%E;wg<-WV[[g]]
    if(!all(dim(wg)==c(dims[g],dims[g])))stop("Moment weight dimensions mismatch")
    if(max(abs(wg-diag(diag(wg))))>1e-10)stop("Full WLS weights outside scope")
    DWs[[g]]<-(ng[g]/N)*t(D)%*%wg
    bread<-bread+DWs[[g]]%*%D
  }
  bread_condition<-kappa(bread,exact=TRUE)
  if(!is.finite(bread_condition)||bread_condition>1e12)stop("Near singular model information")
  K<-E%*%solve(bread,do.call(cbind,DWs))
  # Unnormalized parameter IF rows sum to zero overall; joint household sums
  # retain covariance between responses belonging to different hearing groups.
  psi<-IF%*%t(K)
  h<-unique(d[,c(cluster,stratum),drop=FALSE]);if(anyDuplicated(h[[cluster]]))stop("A PSU belongs to multiple strata")
  Hpsi<-rowsum(psi,group=as.character(d[[cluster]]),reorder=FALSE)
  hhorder<-rownames(Hpsi);st<-as.character(h[[stratum]][match(hhorder,as.character(h[[cluster]]))])
  V<-matrix(0,npar,npar);stratum_n<-table(st)
  if(any(stratum_n<2L))stop("Singleton stratum: no automatic variance adjustment")
  for(z in names(stratum_n)){
    v<-Hpsi[st==z,,drop=FALSE];H<-nrow(v)
    v<-sweep(v,2,colMeans(v),"-");V<-V+crossprod(v)*(if(finite)H/(H-1) else 1)
  }
  V<-(V+t(V))/2;dimnames(V)<-list(names(lavaan::coef(fit)),names(lavaan::coef(fit)))
  out <- list(vcov=V,diagnostics=list(n=N,groups=ng,households=nrow(h),strata=length(stratum_n),
     household_counts=as.list(stratum_n),free_parameters=npar,unique_parameters=nuniq,
     moment_count=sum(dims),max_weighted_moment_if_mean=max(abs(unlist(moment_center))),
     max_margin_error=margin_error,min_bivariate_probability=min_cell,bread_condition=bread_condition,
     finite_correction=finite,cross_group_households=sum(tapply(as.character(d[[group]]),d[[cluster]],function(x)length(unique(x)))>1),
     method="binary_marginal_tetrachoric_IF_stratum_centered_PSU_DWLS_Gauss_Newton"))
  if(debug)out$debug<-list(moment_IF=IF,K=K,unique_map=E,parameter_IF=psi)
  out
}

# Independent engineering verification; all records generated in memory by the
# companion synthetic engine. lavaan internal calls are a reference check only,
# never used by cluster_sandwich itself. Verified against lavaan 0.7-3.
validate_cluster_sandwich <- function() {
  cases<-data.frame(seed=c(21L,22L,23L,24L),weighted=c(FALSE,TRUE,TRUE,FALSE),
     clustered=c(FALSE,FALSE,TRUE,FALSE),rho=c(.85,.85,.85,.95))
  results<-list()
  for(k in seq_len(nrow(cases))){
    cc<-cases[k,];a<-if(k==4)c(4,5,2,8) else anchor_default
    x<-make_synthetic(n=4380,seed=cc$seed,weighted=cc$weighted,clustered=cc$clustered,rho=cc$rho)
    d<-x$data;if(!cc$clustered)d$region<-"one"
    f<-fit_anchored(d,anchors=a);if(!f$ok)stop("Reference fit failed")
    ff<-f$fit;z<-cluster_sandwich(ff,d,finite=FALSE,debug=TRUE)
    oo<-ff@Options;oo$information.bread<-"expected"
    ref<-get("lav_model_nvcov_robust_sem",asNamespace("lavaan"))(
       lavmodel=ff@Model,lavsamplestats=ff@SampleStats,lavdata=ff@Data,
       lavcache=ff@Cache,lavimplied=ff@implied,lavh1=ff@h1,lavoptions=oo,
       lavpartable=ff@ParTable)/nrow(d)
    ds<-d;ds$w<-ds$w*13;zscale<-cluster_sandwich(ff,ds,finite=FALSE)
    set.seed(100);dr<-d[sample.int(nrow(d)),];zrow<-cluster_sandwich(ff,dr,finite=FALSE)
    rel<-function(v,u)max(abs(v-u))/max(abs(u))
    gamma<-lavaan::lavInspect(ff,"gamma");ng<-lavaan::lavInspect(ff,"nobs")
    G<-matrix(0,72,72);G[1:36,1:36]<-gamma[[1]]/ng[1];G[37:72,37:72]<-gamma[[2]]/ng[2]
    row<-data.frame(seed=cc$seed,weighted=cc$weighted,clustered=cc$clustered,rho=cc$rho,
      covariance_relative_error_to_expected_lavaan=rel(z$vcov,ref),
      moment_gamma_relative_error=rel(crossprod(z$debug$moment_IF),G),
      weight_scale_relative_error=rel(zscale$vcov,z$vcov),
      row_order_relative_error=rel(zrow$vcov,z$vcov),
      min_covariance_eigenvalue=min(eigen(z$vcov,symmetric=TRUE,only.values=TRUE)$values),
      cross_group_households=z$diagnostics$cross_group_households,
      covariance_change_if_cross_group_households_split=NA_real_)
    if(cc$clustered){dd<-d;dd$hh<-paste(dd$hh,dd$O,sep="_")
      zz<-cluster_sandwich(ff,dd,finite=FALSE)
      row$covariance_change_if_cross_group_households_split<-rel(zz$vcov,z$vcov)
    }
    if(!cc$clustered)stopifnot(row$covariance_relative_error_to_expected_lavaan<1e-6)
    stopifnot(row$moment_gamma_relative_error<1e-6,row$weight_scale_relative_error<1e-9,
       row$row_order_relative_error<1e-9,row$min_covariance_eigenvalue> -1e-8)
    results[[k]]<-row
  }
  do.call(rbind,results)
}

# Regression for normalized-weight nobs roundoff; deliberately changed row
# counts and group membership must still fail. No estimator formula is altered.
validate_cluster_count_roundoff <- function() {
  results<-list()
  seeds<-c(20271060L,20281063L,20301022L)
  for(k in seq_along(seeds)){
    x<-make_synthetic(seed=seeds[k],clustered=k!=1,weighted=k!=1,
       dif=if(k==3)"uniform" else "null")
    d<-x$data;f<-fit_anchored(d);stopifnot(f$ok)
    ng<-as.numeric(lavaan::lavInspect(f$fit,"nobs"));gl<-lavaan::lavInspect(f$fit,"group.label")
    actual<-vapply(gl,function(g)sum(as.character(d$O)==g),1L)
    z<-cluster_sandwich(f$fit,d)
    missing_one<-d[-1,,drop=FALSE]
    total_error<-tryCatch({cluster_sandwich(f$fit,missing_one);"NO ERROR"},error=function(e)conditionMessage(e))
    moved_one<-d;moved_one$O[1]<-if(as.character(moved_one$O[1])=="O0")"O1" else "O0"
    group_error<-tryCatch({cluster_sandwich(f$fit,moved_one);"NO ERROR"},error=function(e)conditionMessage(e))
    stopifnot(identical(total_error,"Complete-case input must exactly match fitted records"),
       identical(group_error,"Fitted group counts do not match"),all(is.finite(z$vcov)))
    results[[k]]<-data.frame(seed=seeds[k],max_group_count_roundoff=max(abs(ng-actual)),
       total_count_roundoff=sum(ng)-nrow(d),valid_fit_accepted=TRUE,
       missing_record_rejected=TRUE,changed_group_rejected=TRUE)
  }
  do.call(rbind,results)
}
