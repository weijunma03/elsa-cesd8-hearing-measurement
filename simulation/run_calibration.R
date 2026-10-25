source("work/elsa-dif-calibration-2026-10-21/dif_engine.R")
source(file.path(engine_dir,"cluster_sandwich.R"))
args<-commandArgs(TRUE);scenario_id<-args[1]
plan<-read_json(file.path(engine_dir,"calibration_plan.json"),simplifyVector=FALSE)
si<-which(vapply(plan$scenarios,function(x)x$id,"")==scenario_id)
stopifnot(length(si)==1L)
s<-plan$scenarios[[si]]; reps<-if(length(args)>=2)as.integer(args[2]) else plan$repetitions
out<-file.path(engine_dir,"results",scenario_id);dir.create(out,recursive=TRUE,showWarnings=FALSE)
allp<-list();allr<-list();count<-0L;t0<-proc.time()[3]
for(k in seq_len(reps)){
 seed<-20261021L+10000L*si+k
 gen<-make_synthetic(n=s$n,rho=s$rho,clustered=s$clustered,weighted=s$weighted,dif=s$dif,seed=seed)
 d<-gen$data;f<-fit_anchored(d)
 ok<-f$ok;err<-if(is.null(f$error))"" else f$error;cs<-NULL
 if(ok){
  cs<-tryCatch(cluster_sandwich(f$fit,d),error=function(e)e)
  if(inherits(cs,"error")){ok<-FALSE;err<-conditionMessage(cs)}
 }
 if(ok){
  ans<-tryCatch({
   z0<-get_contrasts(f$fit);z0$method<-"ordinary_robust"
   z1<-get_contrasts(f$fit,V=cs$vcov);z1$method<-"household_sandwich"
   z<-rbind(z0,z1);stopifnot(all(is.finite(z$estimate)),all(is.finite(z$se)),all(z$se>0))
   z$truth<-vapply(seq_len(nrow(z)),function(i){r<-gen$truth[gen$truth$item==z$item[i],];if(z$parameter[i]=="a")r$da else r$dt},0)
   z$covered<-abs(z$estimate-z$truth)<=qnorm(.975)*z$se
   z$reject<-z$p_holm<.05;z$rep<-k;z$scenario<-scenario_id
   z
  },error=function(e)e)
  if(inherits(ans,"error")){ok<-FALSE;err<-conditionMessage(ans)} else {count<-count+1L;allp[[count]]<-ans}
 }
 b<-if(f$ok)coef(f$fit) else numeric()
 allr[[k]]<-data.frame(scenario=scenario_id,rep=k,seed=seed,ok=ok,n=nrow(d),households=length(unique(d$hh)),O1_n=sum(d$O=="O1"),floor=mean(rowSums(d[,items])==0),
  mean_affect=if("mA"%in%names(b))unname(b["mA"]) else NA_real_,mean_somatic=if("mS"%in%names(b))unname(b["mS"]) else NA_real_,
  variance_affect=if("vA"%in%names(b))unname(b["vA"]) else NA_real_,variance_somatic=if("vS"%in%names(b))unname(b["vS"]) else NA_real_,
  warning=paste(f$warnings,collapse=" | "),error=err)
 if(k%%50==0L || k==reps){
  write.csv(do.call(rbind,allr),file.path(out,"replicates.csv"),row.names=FALSE)
  if(count>0)write.csv(do.call(rbind,allp),file.path(out,"contrasts.csv"),row.names=FALSE)
  write_json_safe(list(scenario=scenario_id,completed=k,total=reps,successful=count,elapsed_seconds=proc.time()[3]-t0),file.path(out,"progress.json"))
  cat(scenario_id,k,"/",reps,"ok",count,"elapsed",round(proc.time()[3]-t0),"\n");flush.console()
 }
}
write_json_safe(list(scenario=scenario_id,completed=reps,successful=count,elapsed_seconds=proc.time()[3]-t0,
 lavaan=as.character(packageVersion("lavaan")),synthetic_only=TRUE,plan_md5=unname(tools::md5sum(file.path(engine_dir,"calibration_plan.json"))),
 engine_md5=unname(tools::md5sum(file.path(engine_dir,"dif_engine.R"))),covariance_md5=unname(tools::md5sum(file.path(engine_dir,"cluster_sandwich.R")))),file.path(out,"manifest.json"))
