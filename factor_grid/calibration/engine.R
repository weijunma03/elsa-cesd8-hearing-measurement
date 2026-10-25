# Synthetic-only conditional common-reference probability contrast calibration.
# No ELSA participant data are read by this module.
root <- normalizePath(getwd(),winslash="/",mustWork=TRUE)
stage <- file.path(root,"work/elsa-variance-adapter-2026-10-21")
.libPaths(c(file.path(root,"work/elsa-hearing-measurement-2026-10-20/r-library"),.libPaths()))
suppressPackageStartupMessages(library(lavaan)); suppressPackageStartupMessages(library(jsonlite))
anchor_items <- paste0("i",1:8); anchor_affect <- c(1L,4L,5L,6L,7L); anchor_somatic <- c(2L,3L,8L)
anchor_rank_relative_tolerance <- 1e-8
# Reuse frozen model definitions without evaluating their command-line launcher.
oldsyntax <- parse(file.path(root,"work/elsa-anchor-structure-2026-10-21/fit_anchor_structure.R"))
for(e in oldsyntax) if(is.call(e)&&identical(e[[1]],as.name("<-"))&&
  as.character(e[[2]])%in%c("anchor_definitions","anchor_syntax","anchor_identification_diagnostics","anchor_scalar_parameter")) eval(e)
anchors <- c(1L,4L,2L,3L)
lambda_standard <- c(.88,.85,.58,.85,.75,.85,.82,.80)
slope_base <- lambda_standard/sqrt(1-lambda_standard^2)
target_prevalence <- c(.10,.16,.40,.09,.10,.08,.16,.17)
threshold_base <- qnorm(1-target_prevalence)/sqrt(1-lambda_standard^2)
cells <- c("M2_null","M2_mixed","M2_high","M1R_null","M1R_mixed","M1R_high",
           "M2_wrongA","M2_wrongS","M1R_wrongA","M1R_wrongS")
cell_reps <- c(rep(500L,6),rep(250L,4))
writej <- function(x,path) write_json(x,path,auto_unbox=TRUE,pretty=TRUE,na="null",null="null",digits=15)

# RNG state is explicitly supplied; no seed choice after looking at a fit.
generate_domain <- function(cell,stream,n=8000L) {
  stopifnot(cell%in%cells); assign(".Random.seed",stream,envir=.GlobalEnv)
  structure <- strsplit(cell,"_",fixed=TRUE)[[1]][1]
  scenario <- strsplit(cell,"_",fixed=TRUE)[[1]][2]
  H<-round(n/1.27244); sizes<-rep(1L,H); sizes[sample.int(H,n-H)]<-2L
  hi<-rep(seq_len(H),sizes); hh<-sprintf("SYN_H%05d",hi)
  region_h<-as.character(rep(1:9,length.out=H))
  gh<-rbinom(H,1,.2813);g<-gh[hi];independent<-runif(n)<.25
  g[independent]<-rbinom(sum(independent),1,.2813);g[1:2]<-0:1
  if(structure=="M2") {
    rho<-if(scenario=="high").95 else .85
    R<-matrix(c(1,rho,rho,1),2,2)
    eh<-matrix(rnorm(H*2),H,2)%*%chol(R); ee<-matrix(rnorm(n*2),n,2)%*%chol(R)
    eta<-sqrt(.30)*eh[hi,,drop=FALSE]+sqrt(.70)*ee
    eta[g==1,1]<-.30+sqrt(1.20)*eta[g==1,1]
    eta[g==1,2]<-.20+sqrt(1.10)*eta[g==1,2]
    eta_item<-eta[,rep(1,8),drop=FALSE];eta_item[,anchor_somatic]<-eta[,rep(2,3),drop=FALSE]
  } else {
    eh<-rnorm(H);ee<-rnorm(n);eta<-sqrt(.30)*eh[hi]+sqrt(.70)*ee
    eta[g==1]<-.30+sqrt(1.20)*eta[g==1]
    eta_item<-matrix(eta,n,8)
  }
  a<-rbind(slope_base,slope_base);tt<-rbind(threshold_base,threshold_base)
  if(scenario=="mixed"){tt[2,7]<-tt[2,7]+.35;a[2,8]<-a[2,8]*1.30}
  if(scenario=="wrongA")tt[2,1]<-tt[2,1]+.35
  if(scenario=="wrongS")tt[2,2]<-tt[2,2]+.35
  eps<-matrix(rnorm(n*8),n,8)
  if(structure=="M1R") {
    r46<-if(scenario=="high").75 else .35
    eps[,6]<-r46*eps[,4]+sqrt(1-r46*r46)*eps[,6]
  }
  Y<-1L*(a[g+1,,drop=FALSE]*eta_item+eps>tt[g+1,,drop=FALSE])
  d<-as.data.frame(Y);names(d)<-anchor_items
  d$w<-exp(rnorm(H,0,.6))[hi];d$w<-d$w/mean(d$w)
  d$hh<-hh;d$region<-region_h[hi];d$O<-factor(ifelse(g==0,"O0","O1"),levels=c("O0","O1"))
  partition_h<-rbinom(H,1,.40);complete<-rbinom(n,1,.92)
  domain<-partition_h[hi]==1&complete==1
  skeleton<-data.frame(hh=sprintf("SYN_H%05d",seq_len(H)),region=region_h,npeople=sizes)
  truth<-data.frame(item=anchor_items,delta=pnorm(-tt[2,]/sqrt(1+a[2,]^2))-pnorm(-tt[1,]/sqrt(1+a[1,]^2)))
  list(data=d[domain,,drop=FALSE],skeleton=skeleton,truth=truth,structure=structure,
       meta=list(base_n=n,base_households=H,domain_n=sum(domain),domain_households=length(unique(hh[domain])),
                 zero_domain_households=H-length(unique(hh[domain])),strata=9,MCAR_only=TRUE))
}

fit_conditional <- function(d,structure,robust=FALSE) {
  warnings<-character()
  result<-withCallingHandlers(tryCatch({
    stopifnot(all(c("O","w",anchor_items)%in%names(d)),nrow(d)>0,
      !anyNA(d[,c("O","w",anchor_items)]),all(is.finite(d$w)&d$w>0),all(as.matrix(d[,anchor_items])%in%c(0,1)))
    d$O<-factor(as.character(d$O),levels=c("O0","O1"))
    f<-cfa(anchor_syntax(structure,anchors),data=d,ordered=anchor_items,group="O",group.label=c("O0","O1"),
      parameterization="theta",std.lv=FALSE,auto.fix.first=FALSE,meanstructure=TRUE,
      sampling.weights="w",sampling.weights.normalization="group",sampling.weights.type="design",estimator="DWLS",
      se=if(robust)"robust.sem" else "none",information.bread="expected",test="none",control=list(iter.max=1500))
    if(!isTRUE(lavInspect(f,"converged"))||!isTRUE(lavInspect(f,"post.check")))stop("Nonconverged or post-check failure")
    p<-parameterEstimates(f,se=FALSE,zstat=FALSE,pvalue=FALSE,ci=FALSE)
    if(any(!is.finite(p$est))||any(p$est[p$op=="=~"]<=0))stop("Invalid parameters or slopes")
    mats<-c(lavInspect(f,"cov.lv"),lavInspect(f,"theta"))
    if(any(vapply(mats,function(m)min(eigen(m,symmetric=TRUE,only.values=TRUE)$values)<=1e-8,TRUE)))stop("Covariance eigenvalue gate")
    std<-standardizedSolution(f,se=FALSE,ci=FALSE);ss<-std$est.std[std$op=="=~"]
    if(any(!is.finite(ss))||any(ss<0|ss>1+1e-6))stop("Standardized loading gate")
    correl<-if(structure=="M2")lavInspect(f,"cor.lv") else lapply(lavInspect(f,"theta"),cov2cor)
    rv<-if(structure=="M2")vapply(correl,function(m)m[1,2],0) else vapply(correl,function(m)m["i4","i6"],0)
    if(any(!is.finite(rv))||any(abs(rv)>=1-1e-8))stop("Correlation boundary gate")
    id<-anchor_identification_diagnostics(f)
    if(!id$jacobian_rank_full||id$independent_parameters!=if(structure=="M2")30L else 28L)stop("Identification gate")
    list(ok=TRUE,fit=f,error="",rank=id$jacobian_rank)
  },error=function(e)list(ok=FALSE,error=conditionMessage(e))),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
  result$warnings<-warnings;result
}

if(sys.nframe()==0L) {
  RNGkind("L'Ecuyer-CMRG","Inversion","Rejection");set.seed(2026102141L)
  streams<-list();s<-.Random.seed
  for(ci in seq_along(cells))for(k in seq_len(cell_reps[ci])) {
    streams[[paste(cells[ci],k,sep="__")]]<-list(data=s,replicate=parallel::nextRNGSubStream(s))
    s<-parallel::nextRNGStream(s)
  }
  saveRDS(streams,file.path(stage,"calibration/streams.rds"),version=3)
  writej(list(master_seed=2026102141L,kind=RNGkind(),outer_streams=length(streams),
    rule="cell order then index; data stream plus nextRNGSubStream for independent replicate sampling; no replacement of failures",
    synthetic_only=TRUE),file.path(stage,"calibration/rng_manifest.json"))
}
