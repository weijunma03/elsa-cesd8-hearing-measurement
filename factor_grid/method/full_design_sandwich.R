# V2 source-only adapter. Fixed-anchor correct-model Gauss-Newton approximation.
# Default matches local lavaan 0.7.3 DWLS objective/gradient using (n_g-1)/N.
# criterion="lavaan_expected" reproduces package expected-covariance convention
# using n_g/N for reference checks; the two conventions are not exactly equal.
# NOT robust to false anchors or structural misspecification. Final case weights
# are held fixed; finite=TRUE is H/(H-1), not a finite population correction.
# Full skeleton must include all design households, including zero-domain PSUs.

# Synthetic calibration support for complete binary ordinal DWLS models only.
# First-order threshold/tetrachoric moment influence functions, grouped by
# household and centered within observed strata. Model derivative uses DWLS's
# expected-information Gauss-Newton form. No claim of complex-survey validity
# beyond independent PSUs within strata and the supplied fixed case weights.

full_design_sandwich <- function(fit,d,skeleton,debug=FALSE,finite=TRUE,
                                 group="O",weight="w",cluster="hh",stratum="region",
                                 criterion=c("objective","lavaan_expected")) {
  stopifnot(inherits(fit,"lavaan"),all(c(group,weight,cluster,stratum)%in%names(d)))
  opt <- lavaan::lavInspect(fit,"options")
  if(opt$estimator!="DWLS")stop("Only DWLS is implemented")
  if(opt$sampling.weights.normalization!="group")stop("Require group-normalized sampling weights")
  if(anyNA(d[,c(group,weight,cluster,stratum)]))stop("Missing design information")
  criterion <- match.arg(criterion)
  if(!all(c(cluster,stratum)%in%names(skeleton)))stop("Full skeleton lacks household/stratum")
  skeleton <- skeleton[,c(cluster,stratum),drop=FALSE]
  if(anyNA(skeleton)||anyDuplicated(as.character(skeleton[[cluster]])))stop("Full skeleton must have one complete row per household")
  di <- match(as.character(d[[cluster]]),as.character(skeleton[[cluster]]))
  if(anyNA(di))stop("Domain household absent from full skeleton")
  if(any(as.character(d[[stratum]])!=as.character(skeleton[[stratum]][di])))stop("Domain stratum disagrees with full skeleton")
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
  group_factor <- if(criterion=="objective") (ng-1)/N else ng/N
  if(any(!is.finite(group_factor))||any(group_factor<=0))stop("Invalid group criterion factors")
  constrained_D <- do.call(rbind,lapply(delta,function(z)z%*%E))
  singular <- svd(constrained_D,nu=0,nv=0)$d
  rank <- sum(singular>max(singular)*1e-8)
  if(rank!=nuniq)stop("Constrained model moment Jacobian rank deficient")
  for(g in seq_along(gl)) {
    D<-delta[[g]]%*%E;wg<-WV[[g]]
    if(!all(dim(wg)==c(dims[g],dims[g])))stop("Moment weight dimensions mismatch")
    if(max(abs(wg-diag(diag(wg))))>1e-10)stop("Full WLS weights outside scope")
    DWs[[g]]<-group_factor[g]*t(D)%*%wg
    bread<-bread+DWs[[g]]%*%D
  }
  bread_condition<-kappa(bread,exact=TRUE)
  if(!is.finite(bread_condition)||bread_condition>1e12)stop("Near singular model information")
  K<-E%*%solve(bread,do.call(cbind,DWs))
  # Unnormalized parameter IF rows sum to zero overall; joint household sums
  # retain covariance between responses belonging to different hearing groups.
  psi<-IF%*%t(K)
  # Aggregate O0 and O1 together before full-frame stratum centering.
  domain_sums <- rowsum(psi,group=as.character(d[[cluster]]),reorder=FALSE)
  Hpsi <- matrix(0,nrow(skeleton),npar)
  Hpsi[match(rownames(domain_sums),as.character(skeleton[[cluster]])),] <- domain_sums
  st <- as.character(skeleton[[stratum]])
  domain_hh <- unique(as.character(d[[cluster]]))
  V<-matrix(0,npar,npar);stratum_n<-table(st)
  if(any(stratum_n<2L))stop("Singleton stratum: no automatic variance adjustment")
  for(z in names(stratum_n)){
    v<-Hpsi[st==z,,drop=FALSE];H<-nrow(v)
    v<-sweep(v,2,colMeans(v),"-");V<-V+crossprod(v)*(if(finite)H/(H-1) else 1)
  }
  V<-(V+t(V))/2;dimnames(V)<-list(names(lavaan::coef(fit)),names(lavaan::coef(fit)))
  out <- list(vcov=V,diagnostics=list(n=N,groups=ng,households=nrow(skeleton),domain_households=length(domain_hh),
     zero_domain_households=nrow(skeleton)-length(domain_hh),strata=length(stratum_n),
     design_degrees_freedom=nrow(skeleton)-length(stratum_n),jacobian_rank=rank,
     household_counts=as.list(stratum_n),free_parameters=npar,unique_parameters=nuniq,
     moment_count=sum(dims),max_weighted_moment_if_mean=max(abs(unlist(moment_center))),
     max_margin_error=margin_error,min_bivariate_probability=min_cell,bread_condition=bread_condition,
     finite_correction=finite,cross_group_households=sum(tapply(as.character(d[[group]]),d[[cluster]],function(x)length(unique(x)))>1),
     method="full_frame_binary_moment_IF_household_GOR_DWLS_expected_bread",
     adapter_version="v2_objective_consistent",criterion=criterion,group_factor=group_factor,
     approximation="Correct conditional model; fixed weights; WR no FPC; pointwise smooth target",
     misspecification_robust=FALSE))
  if(debug)out$debug<-list(moment_IF=IF,K=K,unique_map=E,parameter_IF=psi,household_IF=Hpsi,household_stratum=st,constrained_jacobian=constrained_D,bread=bread,DWs=DWs)
  out
}


# Probability contrast under common N(0,1), not group-specific latent prevalence.
delta_contrasts <- function(fit,V,anchors=c(1L,4L,2L,3L)) {
  b <- lavaan::coef(fit);p <- lavaan::parTable(fit)
  if(!is.matrix(V)||!all(dim(V)==length(b))||any(!is.finite(V)))stop("Invalid parameter covariance")
  targets <- setdiff(1:8,anchors);G<-matrix(0,length(targets),length(b));out<-vector("list",length(targets))
  for(jj in seq_along(targets)) {
    j<-targets[jj]
    idx<-function(kind,g) {
      lab<-paste0(kind,g,"_",j);ii<-which(p$label==lab & p$free>0)
      if(length(ii)!=1L)stop("Target label absent or nonunique")
      p$free[ii]
    }
    a0<-idx("a",0);a1<-idx("a",1);t0<-idx("t",0);t1<-idx("t",1)
    prob<-function(a,t)pnorm(-t/sqrt(1+a*a))
    deriv<-function(a,t) {
      den<-sqrt(1+a*a);phi<-dnorm(-t/den)
      c(da=phi*t*a/(den^3),dt=-phi/den)
    }
    g0<-deriv(b[a0],b[t0]);g1<-deriv(b[a1],b[t1])
    G[jj,c(a0,t0,a1,t1)]<-c(-g0,g1)
    est<-prob(b[a1],b[t1])-prob(b[a0],b[t0])
    vr<-as.numeric(G[jj,,drop=FALSE]%*%V%*%G[jj,])
    if(!is.finite(vr)||vr< -1e-12)stop("Invalid delta variance")
    out[[jj]]<-data.frame(item=paste0("i",j),estimate=est,se=sqrt(max(vr,0)))
  }
  ans<-do.call(rbind,out);attr(ans,"gradient")<-G;attr(ans,"vcov")<-G%*%V%*%t(G);ans
}
