#train_funcs.R
loadLibs <- function(){
	library(BBmisc)
	library(ggplot2)
	library(ParamHelpers)
	library(stringi)
	library(mlr)
	library(randomForest)
	library(FSelector)
	library(mvtnorm)
	library(modeltools)
	library(sandwich)
	library(strucchange)
	library(party)
}

loadData <- function(dataType){
	fn <- paste("./Sanger-Normalized-DS-",dataType,".RData",sep="")
	load(fn)
	return(data)
}

loadIndex <- function(){
	load("./index.RData")
	message("sample stats")
	message("n training")
	message(length(train.idx))
	message("n validation")
	message(length(valid.idx))
	message("n testing")
	message(length(test.idx))
}

getFVmul <- function(data,targets,train.idx,fwMethod='chi.squared'){
  tmpData <- data.frame( data, targets, stringsAsFactors = F)
  #colnames(tmpData)[1:ncol(targets)] <- colnames(targets)

  #message("resampling instances")
  newData <- tmpData[train.idx,]
  weights <- chi.squared(targets~., newData)[,1]
  names(weights) <- colnames(data)
  
  return(weights)
}

getFVList <- function(data,targets,split.idx,fwMethod='chi.squared'){
  fv.list <- list()
  tmpData <- data.frame(targets=targets, data, stringsAsFactors = F)
  for (ii in 1:length(split.idx$tr)){
    print("resampling instances")
	newData <- tmpData[split.idx$tr[[ii]],]
	task = makeClassifTask(id=paste(dataType,cancerType,sep='.'),data=newData,target="targets",positive=cancerType)
	fv = generateFilterValuesData(task, method = fwMethod)
	fv.list[[ii]] <- fv$data[,3]
  }

  return(fv.list)
}

getFV <- function(data,targets,train.idx,fwMethod='chi.squared'){
  tmpData <- data.frame(targets=targets, data, stringsAsFactors = F)

  message("resampling instances")
  newData <- tmpData[train.idx,]
  task = makeClassifTask(id=paste(dataType,cancerType,sep='.'),data=newData,target="targets",positive=cancerType)
  fv = generateFilterValuesData(task, method = fwMethod)

  return(fv$data[,3])
}

getHPlistMul <- function(targets,fv.list){
	nodesizes <- c(1,2,5)
	mtrys <- c(10,20,50,100)
    hp.list <- list()
    ntrees <- c(500,1000,2000 )
	

    nidx <- 1
	for (nodesize in nodesizes){
		#trees to build
		for (mtry in mtrys){	
			for(ntree in ntrees){
				hp.list[[nidx]] <- list(nodesize=nodesize,mtry=mtry,ntree=ntree)
    			nidx <- nidx + 1
			}
    		
    	}
	} 	
    
    return(hp.list)
}

getHPlist <- function(targets,fv.list){
	ns <- min(table(targets))
    ns2 <- round(ns * 0.63 * 0.66) #0.63 of the minimum group, 2/3 validataion data set
    
    rpmx <- floor(log2(max(table(targets)) / min(table(targets))))
    rpn <- seq(0,min(rpmx,4))

    hp.list <- list()
    nidx <- 1
    ntrees <- c(500,1000,2000)
	#positive/negative ds ratio
	for (rt in rpn){
		#trees to build
		for (ntree in ntrees){
    		if (which(table(targets)==max(table(targets)))==1){
    			nss1 <- 2^rt*ns2
    			nss2 <- ns2
    		} else {
    			nss1 <- ns2
    			nss2 <- 2^rt*ns2
    		}
    		
    		hp.list[[nidx]] <- list(nss1=nss1,nss2=nss2,ntree=ntree)
    		nidx <- nidx + 1
    	}
	} 	
    
    return(hp.list)
}

conf2f1 <- function(conf,cls){
	f1 <- 2*conf[cls,cls]/(2*conf[cls,cls]+conf["none",cls]+conf[cls,"none"])
	return(f1)
}

funCV <- function(fv.list,hp.list,data,targets,dataType,cancerType,train.idx,test.idx,valid.idx,cv.idx,idx.map){
	#grid search on a set of hyperparameters

    newData <- data.frame(targets=targets,data,stringsAsFactors = F)


    bestMods <- list()
	bestf1.valid <- 0
	bestf1.test <- 0
	best.prob.ts <- c()
	best.prob.va <- c()
	besthp <- list()
	for (nn in 1:length(hp.list)){
		nss1 <- hp.list[[nn]]$nss1
		nss2 <- hp.list[[nn]]$nss2
		ntree <- hp.list[[nn]]$ntree
		lrn <- makeLearner("classif.randomForest",id="rf.0p6ns", predict.type = "prob",par.vals=list(replace = F, sampsize=c(nss1,nss2), ntree=ntree),importance=T)
        
        cv.valid.f1 <- 0
	    cv.train.f1 <- 0
	    cv.test.f1 <- 0
	    nfld <- length(cv.idx$tr)
	    mod.list <- list()
	    ts.idx <- match(test.idx, rownames(idx.map))
	    pred.p.ts <- c()

	    for (jj in 1:nfld){
	    	fea <- which(fv.list[[jj]]!=0)
        	fea.task <- makeClassifTask(id=paste(dataType,cancerType,sep='.'),data=newData[,c(1,fea+1)],target="targets",positive=cancerType)
        	tr.idx <- match(cv.idx$tr[[jj]],rownames(idx.map))
			va.idx <- match(cv.idx$ts[[jj]],rownames(idx.map))

	        mod.list[[jj]] = train(lrn,fea.task,subset = tr.idx)


	        valid.pred = predict(mod.list[[jj]], task = fea.task, subset = va.idx)
	        test.pred = predict(mod.list[[jj]], task = fea.task, subset = ts.idx)
	        

	        f1.train <- conf2f1(mod.list[[jj]]$learner.model$confusion,cancerType)

	        f1.valid <- performance(valid.pred,measures = f1)
	        f1.test <- performance(test.pred,measures = f1)
	        cv.train.f1 <- cv.train.f1 + f1.train
	        cv.valid.f1 <- cv.valid.f1 + f1.valid
	        cv.test.f1 <- cv.test.f1 + f1.test
	        ctstr <- paste("prob",cancerType,sep=".")

	        if (is.null(pred.p.ts)){
	        	pred.p.ts <- test.pred$data[,ctstr]
				pred.p.va <- valid.pred$data[,ctstr]
        	} else{
        		pred.p.ts <- pred.p.ts + test.pred$data[,ctstr]
				pred.p.va <- pred.p.va + valid.pred$data[,ctstr]
        	}
	    }
	    
	    f1.valid.curr <- cv.valid.f1/3
	    if (f1.valid.curr > bestf1.valid){
	    	bestf1.train <- cv.train.f1/3
	    	bestf1.valid <- f1.valid.curr
	    	bestf1.test <- cv.test.f1/3
	    	bestMods <- mod.list
	    	besthp <- hp.list[[nn]]
	    	best.prob.ts <- pred.p.ts/nfld
			best.prob.va <- pred.p.va/nfld
	    	message(paste("updated best f1 on validataion",f1.valid.curr))

	    message(paste("nss1",besthp$nss1,"nss2",besthp$nss2,"ntree",besthp$ntree))
	    }
	}


	return(list(bestf1.train=bestf1.train,bestf1.valid=bestf1.valid,bestf1.test=bestf1.test,bestMods=bestMods,besthp=besthp,best.prob.ts=best.prob.ts,best.prob.va=best.prob.va))
	
}

oneValid <- function(fv,hp.list,data,targets,dataType,cancerType,train.idx,test.idx,valid.idx,cv.idx,idx.map){
	#grid search on a set of hyperparameters

    newData <- data.frame(targets=targets,data,stringsAsFactors = F)


    bestMods <- list()
	bestf1.train <- 0
	bestf1.valid <- 0
	bestf1.test <- 0
	best.prob.ts <- c()
	best.prob.va <- c()
	besthp <- list()
	for (nn in 1:length(hp.list)){
		nss1 <- hp.list[[nn]]$nss1
		nss2 <- hp.list[[nn]]$nss2
		ntree <- hp.list[[nn]]$ntree
		lrn <- makeLearner("classif.randomForest",id="rf.0p6ns", predict.type = "prob",par.vals=list(replace = F, sampsize=c(nss1,nss2), ntree=ntree),importance=T)
        
        valid.f1 <- 0
	    train.f1 <- 0
	    test.f1 <- 0
	    ts.idx <- match(test.idx, rownames(idx.map))
	    tr.idx <- match(train.idx,rownames(idx.map))
	    va.idx <- match(valid.idx,rownames(idx.map))

    	fea <- which(fv!=0)
    	fea.task <- makeClassifTask(id=paste(dataType,cancerType,sep='.'),data=newData[,c(1,fea+1)],target="targets",positive=cancerType)


        mod = train(lrn,fea.task,subset = tr.idx)

        valid.pred = predict(mod, task = fea.task, subset = va.idx)
        test.pred = predict(mod, task = fea.task, subset = ts.idx)

        f1.train <- conf2f1(mod$learner.model$confusion,cancerType)

        f1.valid <- performance(valid.pred,measures = f1)
        f1.test <- performance(test.pred,measures = f1)
        ctstr <- paste("prob",cancerType,sep=".")
		pred.p.va <- valid.pred$data[,ctstr]
		pred.p.ts <- test.pred$data[,ctstr]

	    if (f1.valid > bestf1.valid){
	    	bestf1.train <- f1.train
	    	bestf1.valid <- f1.valid
	    	bestf1.test <- f1.test
	    	bestMods <- mod
	    	besthp <- hp.list[[nn]]
	    	best.prob.ts <- pred.p.ts
	    	best.prob.va <- pred.p.va
	    	message(paste("updated best f1 on validataion",bestf1.valid))

	        message(paste("nss1",besthp$nss1,"nss2",besthp$nss2,"ntree",besthp$ntree))
	    }
	}


	return(list(bestf1.train=f1.train,bestf1.valid=bestf1.valid,bestf1.test=bestf1.test,bestMods=bestMods,besthp=besthp,best.prob.ts=best.prob.ts,best.prob.va=best.prob.va))
	
}


mergeData <- function(dtStr){
  dataTypes <- strsplit(dtStr,"+",fixed=T)
  tmpData <- c()

  for (dataType in dataTypes[[1]]){
    #mergeData
    dfn <- paste("/u/wjiao/DataSets/Pan-Cancer/data-feb-2017/Sanger-Normalized-DS-",dataType,".RData",sep="")
    load(dfn)
    if(is.null(tmpData)){
      tmpData <- data
    }else{
      tmpData <- cbind(tmpData,data)
    }
  }
  return(tmpData)
}

mconf2f1 <- function(conf){
	int.cls <- intersect(colnames(conf), rownames(conf))
	return(sum(diag(conf[int.cls,int.cls]))/sum(conf[int.cls,int.cls]))

}

oneValidMul <- function(fv,hp.list,data,targets,dataType,cancerType,train.idx,test.idx,valid.idx,cv.idx,idx.map){
	#grid search on a set of hyperparameters

	# biTargets <- makeMulTargets(idx.map)
	# tarNames <- gsub("-",".",colnames(biTargets))

	coln <- names(fv[which(fv!=0)])
	newData <- data.frame(targets=idx.map[,2],data[,coln],stringsAsFactors = F)



    bestMods <- list()
	bestf1.train <- 0
	bestf1.valid <- 0
	bestf1.test <- 0
	best.prob.va <- c()
	best.prob.ts <- c()
	best.conf.va <- c()
	best.conf.ts <- c()
	besthp <- list()
	for (nn in 1:length(hp.list)){
		mtry <- hp.list[[nn]]$mtry
		nodesize <- hp.list[[nn]]$nodesize
		ntree <- hp.list[[nn]]$ntree
		lrn <- makeLearner("classif.randomForest",id="rf.0p6ns", predict.type = "prob",par.vals=list(mtry=mtry, nodesize=nodesize, ntree=ntree),importance=T)

        
        valid.f1 <- 0
	    train.f1 <- 0
	    test.f1 <- 0
	    ts.idx <- match(test.idx, rownames(idx.map))
	    tr.idx <- match(train.idx,rownames(idx.map))
	    va.idx <- match(valid.idx,rownames(idx.map))

		fea.task <- makeClassifTask(id=paste(dataType,cancerType,sep='.'),data=newData,target="targets")


        mod = train(lrn,fea.task,subset = tr.idx)


        pred.p.va = predict(mod$learner.model, task=fea.task, newdata = newData[va.idx,], type = "prob")
		pred.p.ts = predict(mod$learner.model, task=fea.task, newdata = newData[ts.idx,], type = "prob")
		valid.pred.lbl = predict(mod$learner.model, task=fea.task, newdata = newData[va.idx,])
		test.pred.lbl = predict(mod$learner.model, task=fea.task, newdata = newData[ts.idx,])
        

        f1.train <- mconf2f1(mod$learner.model$confusion)
        conf.va <- table(valid.pred.lbl,idx.map[va.idx,2])
        conf.ts <- table(test.pred.lbl,idx.map[ts.idx,2])
        f1.valid <- mconf2f1(conf.va)
        f1.test <- mconf2f1(conf.ts)
        
	    if (f1.valid > bestf1.valid){
	    	bestf1.train <- f1.train
	    	bestf1.valid <- f1.valid
	    	bestf1.test <- f1.test
	    	bestMods <- mod
	    	besthp <- hp.list[[nn]]
	    	best.prob.ts <- pred.p.ts
	    	best.prob.va <- pred.p.va
	    	best.conf.ts <- conf.ts
	    	best.conf.va <- conf.va
	    	message(paste("updated best f1 on validataion",bestf1.valid))

	        message(paste("nss1",besthp$nss1,"nss2",besthp$nss2,"ntree",besthp$ntree))
	    }
	}


	return(list(bestf1.train=f1.train,bestf1.valid=bestf1.valid,bestf1.test=bestf1.test,bestMods=bestMods,besthp=besthp,best.prob.ts=best.prob.ts,best.prob.va=best.prob.va,best.conf.ts=best.conf.ts,best.conf.va=best.conf.va))
	
}

