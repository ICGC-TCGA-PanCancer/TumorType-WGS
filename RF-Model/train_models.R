#run LR with SNV
args <- commandArgs()
print(args)
#args <- c("MUT","Breast-AdenoCA","chi.squared")
dataType <- args[0]
cancerType <- args[1]
fwMethod <- args[2]

source("./train_funcs.R")

print ("load required libraries ...")
loadLibs()

print ("load data ...")
data <- mergeData(dataType)

print("load index")
load("./index.RData")

##make resampling instances
targets <- idx.map[,2]

#get fvlist
fv <- getFVmul(data,targets,train.idx)

#get hyper parameter list
hp.list <- getHPlistMul(targets,fv)

#run CV and selet models with the best hyper parameter setting
resultList <- oneValidMul(fv,hp.list,data,targets,dataType,cancerType,train.idx,test.idx,valid.idx,cv.idx,idx.map)

print("save data")
tryCatch({
save(resultList,file=paste("RF.Models.",dataType,cancerType,fwMethod,"RData",sep='.'))
print("dataSaved")
},error = function(e){
  print("Saving data error") 
})


