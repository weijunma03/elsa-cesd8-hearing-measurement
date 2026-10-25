# Synthetic-only calibration. Explicit probit response-function scale, residual variance 1.
engine_dir <- normalizePath(file.path(getwd(), "work/elsa-dif-calibration-2026-10-21"), winslash="/", mustWork=TRUE)
.libPaths(c(file.path(dirname(engine_dir),"elsa-hearing-measurement-2026-10-20/r-library"),.libPaths()))
suppressPackageStartupMessages(library(lavaan))
suppressPackageStartupMessages(library(jsonlite))
items <- paste0("i",1:8)
affect <- c(1,4,5,6,7); somatic <- c(2,3,8)
anchor_default <- c(1,4,2,3) # SIMULATED oracle anchors; not recommendations for ELSA.
lambda_standard <- c(.88,.85,.58,.85,.75,.85,.82,.80)
slope_base <- lambda_standard/sqrt(1-lambda_standard^2)
target_prevalence <- c(.10,.16,.40,.09,.10,.08,.16,.17)
threshold_base <- qnorm(1-target_prevalence)/sqrt(1-lambda_standard^2)
write_json_safe <- function(x,path) write_json(x,path,auto_unbox=TRUE,pretty=TRUE,na="null",null="null",digits=12)

make_synthetic <- function(n=4380L,rho=.85,clustered=TRUE,weighted=TRUE,dif="null",seed=20261021L,p1=.2813) {
 set.seed(seed)
 nh <- if(clustered)round(n/1.27244) else n
 sizes <- rep(1L,nh)
 if(n>nh)sizes[sample.int(nh,n-nh)]<-2L
 hh <- rep(seq_len(nh),sizes); region_h <- rep(1:9,length.out=nh)
 # Correlated group membership within households; some mixed-group households remain.
 gh <- rbinom(nh,1,p1); g <- gh[hh]
 independent <- runif(n) < .25
 g[independent] <- rbinom(sum(independent),1,p1)
 g[1:2] <- 0:1
 R <- matrix(c(1,rho,rho,1),2,2)
 eta_h <- matrix(rnorm(nh*2),nh,2)%*%chol(R)
 eta_e <- matrix(rnorm(n*2),n,2)%*%chol(R)
 icc <- if(clustered).30 else 0
 eta <- sqrt(icc)*eta_h[hh,,drop=FALSE]+sqrt(1-icc)*eta_e
 eta[g==1,1] <- .30+sqrt(1.20)*eta[g==1,1]
 eta[g==1,2] <- .20+sqrt(1.10)*eta[g==1,2]
 a <- rbind(slope_base,slope_base); t <- rbind(threshold_base,threshold_base)
 if(dif %in% c("uniform","mixed"))t[2,7] <- t[2,7]+.35
 if(dif %in% c("slope","mixed"))a[2,8] <- a[2,8]*1.30
 if(dif=="anchor_contamination")t[2,1] <- t[2,1]+.35
 if(dif=="somatic_contamination")t[2,2] <- t[2,2]+.35
 eta_item <- eta[,rep(1,8),drop=FALSE];eta_item[,somatic]<-eta[,rep(2,3),drop=FALSE]
 eps <- matrix(rnorm(n*8),n,8)
 Y <- 1L*(a[g+1,,drop=FALSE]*eta_item+eps > t[g+1,,drop=FALSE])
 d <- as.data.frame(Y);names(d)<-items
 d$w <- if(weighted)exp(rnorm(nh,0,.6))[hh] else rep(1,n)
 d$w <- d$w/mean(d$w); d$hh <- as.character(hh); d$region <- as.character(region_h[hh])
 d$O <- factor(ifelse(g==0,"O0","O1"),levels=c("O0","O1"))
 truth <- data.frame(item=items,da=a[2,]-a[1,],dt=t[2,]-t[1,])
 list(data=d,truth=truth,latent_truth=c(mA=.30,mS=.20,vA=1.20,vS=1.10,c0=rho,c1=rho*sqrt(1.20*1.10)))
}

anchored_syntax <- function(anchors=anchor_default) {
 lab <- function(kind,j)if(j%in%anchors)paste0("c(",kind,j,",",kind,j,")") else paste0("c(",kind,"0_",j,",",kind,"1_",j,")")
 lines <- c(paste0("Affect =~ ",paste(vapply(affect,function(j)paste0(lab("a",j),"*i",j),""),collapse=" + ")),
            paste0("Somatic =~ ",paste(vapply(somatic,function(j)paste0(lab("a",j),"*i",j),""),collapse=" + ")))
 for(j in 1:8)lines<-c(lines,paste0("i",j," | ",lab("t",j),"*t1"),paste0("i",j," ~~ c(1,1)*i",j),paste0("i",j," ~ c(0,0)*1"))
 lines<-c(lines,'Affect ~~ c(1,NA)*c("","vA")*Affect','Somatic ~~ c(1,NA)*c("","vS")*Somatic','Affect ~ c(0,NA)*c("","mA")*1','Somatic ~ c(0,NA)*c("","mS")*1',"Affect ~~ c(c0,c1)*Somatic")
 paste(lines,collapse="\n")
}

fit_anchored <- function(d,anchors=anchor_default,robust=TRUE) {
 warns<-character()
 fit<-withCallingHandlers(tryCatch(lavaan::cfa(anchored_syntax(anchors),data=d,group="O",group.label=c("O0","O1"),
   ordered=items,parameterization="theta",std.lv=FALSE,auto.fix.first=FALSE,meanstructure=TRUE,
   sampling.weights="w",sampling.weights.normalization="group",estimator="DWLS",
   se=if(robust)"robust.sem" else "none",information.bread="expected",test="none",control=list(iter.max=1000)),error=function(e)e),
   warning=function(w){warns<<-c(warns,conditionMessage(w));invokeRestart("muffleWarning")})
 if(inherits(fit,"error"))return(list(ok=FALSE,error=conditionMessage(fit),warnings=warns))
 ok<-isTRUE(lavInspect(fit,"converged")) && isTRUE(suppressWarnings(lavInspect(fit,"post.check")))
 p<-parameterEstimates(fit); ok<-ok && all(is.finite(p$est)) && all(p$est[p$op=="=~"]>0)
 list(ok=ok,fit=fit,warnings=unique(warns),parameters=p)
}

get_contrasts <- function(fit,anchors=anchor_default,V=NULL) {
 p<-parTable(fit); b<-coef(fit); if(is.null(V))V<-vcov(fit)
 targets<-setdiff(1:8,anchors);out<-list();i<-0L
 for(j in targets)for(kind in c("a","t")){
  l0<-paste0(kind,"0_",j);l1<-paste0(kind,"1_",j)
  k0<-p$free[match(l0,p$label)];k1<-p$free[match(l1,p$label)]
  C<-rep(0,length(b));C[k1]<-1;C[k0]<--1
  est<-sum(C*b);se<-sqrt(as.numeric(t(C)%*%V%*%C))
  i<-i+1L;out[[i]]<-data.frame(item=paste0("i",j),parameter=kind,estimate=est,se=se,z=est/se,p=2*pnorm(-abs(est/se)))
 }
 r<-do.call(rbind,out);r$p_holm<-p.adjust(r$p,"holm");r
}

bootstrap_households <- function(d) {
 h<-unique(d[,c("hh","region")]);stopifnot(!anyDuplicated(h$hh))
 rows<-split(seq_len(nrow(d)),d$hh);res<-list();k<-0L
 for(r in unique(h$region)){
  ids<-h$hh[h$region==r];sampled<-sample(ids,length(ids),replace=TRUE)
  for(id in sampled){k<-k+1L;z<-d[rows[[id]],,drop=FALSE];z$hh<-as.character(k);res[[k]]<-z}
 }
 d<-do.call(rbind,res);rownames(d)<-NULL;d
}
