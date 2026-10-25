# Local fixed full-sample analysis. Participant rows remain in stdin/R memory.
ROOT_REAL <- normalizePath(getwd(),winslash="/",mustWork=TRUE)
REAL_STAGE <- file.path(ROOT_REAL,"work/elsa-real-analysis-2026-10-24")
source(file.path(ROOT_REAL,"work/elsa-variance-adapter-2026-10-21/calibration/engine.R"))
source(file.path(ROOT_REAL,"work/elsa-variance-adapter-2026-10-21/method/full_design_sandwich.R"))
source(file.path(ROOT_REAL,"work/elsa-method-candidates-2026-10-22/interval/studentized_adapter.R"))
plan <- jsonlite::read_json(file.path(REAL_STAGE,"analysis_plan.json"),simplifyVector=TRUE)
freeze <- jsonlite::read_json(file.path(REAL_STAGE,"plan_freeze.json"),simplifyVector=TRUE)
stopifnot(candidate_sha256(file.path(REAL_STAGE,"analysis_plan.json"))==freeze$plan_sha256)
for(p in names(plan$source_hashes))stopifnot(candidate_sha256(file.path(ROOT_REAL,p))==plan$source_hashes[[p]])
stopifnot(as.character(getRversion())=="4.6.1",as.character(packageVersion("lavaan"))=="0.7.3")
out <- file.path(REAL_STAGE,"results")
stopifnot(!dir.exists(out));dir.create(out)
f <- read.csv(file("stdin","r"),stringsAsFactors=FALSE,check.names=FALSE,na.strings="")
stopifnot(identical(names(f),c("hh","region","region_alt","w","domain","partition","O",paste0("i",1:8))))
stopifnot(nrow(f)==8152,sum(f$domain)==7314,length(unique(f$hh))==6217)
sk <- unique(f[,c("hh","region")]);ska <- unique(data.frame(hh=f$hh,region=f$region_alt))
stopifnot(!anyDuplicated(sk$hh),!anyDuplicated(ska$hh),nrow(sk)==6217,
          length(unique(sk$region))==9,min(table(sk$region))>=2,
          sum(f$region!=f$region_alt)==6,length(unique(f$hh[f$region!=f$region_alt]))==3)
d <- f[f$domain==1,,drop=FALSE]; d$O <- factor(d$O,levels=c("O0","O1"))
stopifnot(nrow(d)==7314,length(unique(d$hh))==5748,sum(d$O=="O0")==5241,sum(d$O=="O1")==2073,
          all(as.matrix(d[,anchor_items])%in%c(0,1)),all(is.finite(d$w)&d$w>0),nrow(sk)-length(unique(sk$region))==6208)
writej(list(status="started",planned_fits=60,plan_sha256=freeze$plan_sha256,
            n=7314,frame_n=8152,frame_households=6217,design_df=6208),file.path(out,"started.json"))
fits<-list();points<-list();curves<-list();residuals<-list();loads<-list();cors<-list();intervals<-list();warnings_out<-list()
k<-0L;inference_ok<-FALSE
add_prefix <- function(x,structure,anchor_set){x$structure<-structure;x$anchor_set<-anchor_set;x}
for(structure in c("M2","M1R"))for(def in anchor_definitions()) {
  k<-k+1L; anchors<-as.integer(def$anchors)
  intent<-list(slot=k,structure=structure,anchor_set=def$id,anchors=anchors,plan_sha256=freeze$plan_sha256)
  cat(toJSON(intent,auto_unbox=TRUE,digits=16),"\n",file=file.path(out,"intents.jsonl"),append=TRUE,sep="")
  z<-fit_conditional(d,structure)
  fr<-data.frame(slot=k,structure=structure,anchor_set=def$id,anchors=paste(paste0("i",anchors),collapse=";"),
                 fit_ok=isTRUE(z$ok),rank=if(isTRUE(z$ok))z$rank else NA_integer_,error=z$error)
  if(length(z$warnings))warnings_out[[length(warnings_out)+1L]]<-data.frame(slot=k,structure=structure,anchor_set=def$id,warning=z$warnings)
  if(z$ok) {
    fit<-z$fit;p<-parameterEstimates(fit,se=FALSE,zstat=FALSE,pvalue=FALSE,ci=FALSE)
    point<-delta_contrasts(fit,matrix(0,length(coef(fit)),length(coef(fit))),anchors=anchors)
    pp<-list();cc<-list()
    for(j in setdiff(1:8,anchors)) {
      it<-paste0("i",j);a0<-anchor_scalar_parameter(p,1,it,"=~");a1<-anchor_scalar_parameter(p,2,it,"=~")
      t0<-anchor_scalar_parameter(p,1,it,"|");t1<-anchor_scalar_parameter(p,2,it,"|")
      p0<-pnorm(-t0/sqrt(1+a0*a0));p1<-pnorm(-t1/sqrt(1+a1*a1))
      stopifnot(abs(point$estimate[point$item==it]-(p1-p0))<1e-12)
      coordinate<-if(structure=="M1R")"General" else if(j%in%anchor_affect)"Affect" else "Somatic"
      pp[[length(pp)+1L]]<-data.frame(item=it,coordinate=coordinate,a0=a0,a1=a1,t0=t0,t1=t1,
                       p0_reference=p0,p1_reference=p1,estimate=p1-p0)
      eta<-c(-2,-1,0,1,2)
      cc[[length(cc)+1L]]<-data.frame(item=it,coordinate=coordinate,eta=eta,
                       p0=pnorm(a0*eta-t0),p1=pnorm(a1*eta-t1),estimate=pnorm(a1*eta-t1)-pnorm(a0*eta-t0))
    }
    points[[length(points)+1L]]<-add_prefix(do.call(rbind,pp),structure,def$id)
    curves[[length(curves)+1L]]<-add_prefix(do.call(rbind,cc),structure,def$id)
    st<-standardizedSolution(fit,se=FALSE,ci=FALSE)
    ld<-st[st$op=="=~",c("lhs","rhs","group","est.std")]
    names(ld)<-c("factor","item","group","loading_std")
    ld$group<-c("O0","O1")[ld$group];ld$anchor<-as.integer(sub("i","",ld$item))%in%anchors
    loads[[length(loads)+1L]]<-add_prefix(ld,structure,def$id)
    cv<-if(structure=="M2")lavInspect(fit,"cor.lv") else lapply(lavInspect(fit,"theta"),cov2cor)
    rho<-if(structure=="M2")vapply(cv,function(m)m[1,2],0) else vapply(cv,function(m)m["i4","i6"],0)
    cors[[length(cors)+1L]]<-add_prefix(data.frame(group=c("O0","O1"),rho=as.numeric(rho),
                 type=if(structure=="M2")"Affect_Somatic_factor" else "i4_i6_residual"),structure,def$id)
    obs<-lavInspect(fit,"wls.obs");est<-lavInspect(fit,"wls.est")
    for(g in 1:2){
      stopifnot(length(obs[[g]])==length(est[[g]]))
      residuals[[length(residuals)+1L]]<-add_prefix(data.frame(group=c("O0","O1")[g],
                 moment=names(obs[[g]]),observed=as.numeric(obs[[g]]),implied=as.numeric(est[[g]]),
                 residual=as.numeric(obs[[g]])-as.numeric(est[[g]])),structure,def$id)
    }
    if(structure=="M1R"&&def$id=="AS01") {
      # Same fitted point parameters; only declared household GOR ties differ.
      diag_bag<-list();iv<-list();variance_error<-""
      variance_ok<-tryCatch({
        for(rule in c("lex_first","lex_last")) {
          dd<-d;ss<-sk
          if(rule=="lex_last"){dd$region<-dd$region_alt;ss<-ska}
          cs<-full_design_sandwich(fit,dd,ss,finite=TRUE,criterion="objective")
          an<-delta_contrasts(fit,cs$vcov,anchors=anchors)
          stopifnot(identical(an$item,paste0("i",5:8)),all(is.finite(an$se)&an$se>0),cs$diagnostics$design_degrees_freedom==6208)
          G<-attr(an,"gradient");Vt<-attr(an,"vcov");df<-6208;crit<-qt(.975,df)
          an$df<-df;an$lower<-an$estimate-crit*an$se;an$upper<-an$estimate+crit*an$se
          an$t<-an$estimate/an$se;an$p_two_sided<-2*pt(-abs(an$t),df)
          an$p_holm_four<-p.adjust(an$p_two_sided,method="holm",n=4)
          an$tie_rule<-rule;an$pointwise<-TRUE;an$conditional_exploratory<-TRUE
          iv[[rule]]<-an;diag_bag[[rule]]<-cs$diagnostics
          write.csv(G,file.path(out,paste0("reference_gradient_",rule,".csv")),row.names=FALSE)
          write.csv(cs$vcov,file.path(out,paste0("reference_parameter_vcov_",rule,".csv")),row.names=FALSE)
          write.csv(Vt,file.path(out,paste0("reference_target_vcov_",rule,".csv")),row.names=FALSE)
        }
        TRUE
      },error=function(e){variance_error<<-conditionMessage(e);FALSE})
      writej(list(ok=variance_ok,error=variance_error,diagnostics=diag_bag,
             scope="Correct conditional model with fixed anchors and weights; not robust to misspecification"),file.path(out,"reference_variance_diagnostics.json"))
      if(variance_ok){intervals<-iv;inference_ok<-TRUE}
    }
  }
  fits[[k]]<-fr
  cat(toJSON(list(slot=k,structure=structure,anchor_set=def$id,fit_ok=z$ok,error=z$error),auto_unbox=TRUE,digits=16),
      "\n",file=file.path(out,"results.jsonl"),append=TRUE,sep="")
  writej(list(completed=k,total=60),file.path(out,"progress.json"))
  if(k%%5L==0L)cat(k,"/60 fixed full-sample fits completed\n",sep="")
}
export <- function(x,name){if(length(x))write.csv(do.call(rbind,x),file.path(out,name),row.names=FALSE,na="")}
export(fits,"fits.csv");export(points,"reference_integrals.csv");export(curves,"curves.csv")
export(residuals,"moment_residuals.csv");export(loads,"loadings.csv");export(cors,"correlations.csv")
export(intervals,"conditional_reference_intervals.csv");export(warnings_out,"warnings.csv")
writej(list(status="complete",planned_fits=60,attempted_fits=k,fit_successes=sum(vapply(fits,function(x)x$fit_ok,TRUE)),
            reference_interval_available=inference_ok,plan_sha256=freeze$plan_sha256,
            original_primary_structure="M2",reference_inference_structure="M1R",reference_anchor="AS01",
            n=7314,domain_households=5748,frame_n=8152,frame_households=6217,design_df=6208,
            participant_files_written=0,latent_scores_written=0,
            R_version=as.character(getRversion()),lavaan_version=as.character(packageVersion("lavaan"))),file.path(out,"completion.json"))
