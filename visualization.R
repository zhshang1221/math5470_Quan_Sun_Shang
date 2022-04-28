library(plotrix)
rm(list=ls())

work_path <- '/Users/shangzh/Desktop/22Spring/MATH5470/re-generate'
setwd(work_path)

data_path <- './empirical_result/'

# plot monthly out-of-sample R_oos result
par(mfrow=c(1,2))
monthly_data <- paste(data_path, 'monthly_oos.csv', sep="")
monthly_data_frame <- read.csv(monthly_data, header=TRUE, sep=",",stringsAsFactors=FALSE)
monthly_data <- as.matrix(t(monthly_data_frame[, 2:4]))

colnames(monthly_data) <- c('OLS', 'OLS-3', 'PLS', 'PCR', 'GL', 'BoostTree')
barplot(height=monthly_data, main = "Monthly Out-of-sample Prediction Performance", beside=TRUE, col= cm.colors(3), args.legend=c(35,28, "bottomright"))
legend("bottomright", title="Features Selection", c('All', 'Top 1000', 'Bottom 1000'), x.intersp = 0.7,y.intersp = 1.0,text.width=6,cex=1, fill = cm.colors(3))

# monthly data without OLS
monthly_data <- as.matrix(t(monthly_data_frame[2:6,  2:4]))
colnames(monthly_data) <- c('OLS-3', 'PLS', 'PCR', 'GL', 'BoostTree')
barplot(height=monthly_data, main = "Monthly Out-of-sample Prediction Performance(without OLS)", beside=TRUE, col= cm.colors(3), args.legend=c(35,28, "bottomright"))

# plot annual out-of-sample R_oos result
annual_data <- paste(data_path, 'annual_oos.csv', sep="")
annual_data_frame <- read.csv(annual_data, header=TRUE, sep=",",stringsAsFactors=FALSE)
annual_data <- as.matrix(t(annual_data_frame[, 2:4]))

colnames(annual_data) <- c('OLS', 'OLS-3', 'PLS', 'PCR', 'GL', 'BoostTree')
barplot(height=annual_data, main = "Annual Out-of-sample Prediction Performance", beside=TRUE, col= cm.colors(3), args.legend=c(35,28, "bottomright"))
legend("bottomright", title="Features Selection", c('All', 'Top 1000', 'Bottom 1000'), x.intersp = 0.7,y.intersp = 1.0,text.width=6,cex=1, fill = cm.colors(3))

# annual data without OLS
annual_data <- as.matrix(t(monthly_data_frame[2:6,  2:4]))
colnames(annual_data) <- c('OLS-3', 'PLS', 'PCR', 'GL', 'BoostTree')
barplot(height=annual_data, main = "Annual Out-of-sample Prediction Performance(without OLS)", beside=TRUE, col= cm.colors(3), args.legend=c(35,28, "bottomright"))

# time-varying model complexity
par(mfrow=c(2,2))
complexity_info <- read.csv(paste(data_path, 'model_complexity.csv', sep=""), header=TRUE, sep=",",stringsAsFactors=FALSE)
plot(complexity_info$year, complexity_info$PLS, main='PLS', col="#CD0074",ylab="# of Comp.", xlab="Year", 'l', lwd=2)
plot(complexity_info$year, complexity_info$PCR, main='PCR', col="#CD0074",ylab="# of Comp.", xlab="Year", 'l', lwd=2)
plot(complexity_info$year, complexity_info$GL, main='Generalized Linear', col="#CD0074",ylab="# of Char.", xlab="Year", 'l', lwd=2)
plot(complexity_info$year, complexity_info$BoostTree, main='Boost Tree', col="#CD0074",ylab="# of Char.", xlab="Year", 'l', lwd=2)

# variable importance
par(mfrow=c(2, 2))
variable_importance_info <- read.csv(paste(data_path, 'variable_importance.csv', sep=""), header=TRUE, sep=",",stringsAsFactors=FALSE)
barplot(rev(variable_importance_info$PLS),horiz=T,xlim=c(-0.5,0.5), axes=F,col=rep(cm.colors(1),each=15), main="PLS")
text(seq(from=1.5,length.out=135,by=1.2),x=-0.1, label=rev(variable_importance_info$features))
axis(1,c(0,0.1,0.2,0.3,0.4),c(0,0.1,0.2,0.3,0.4))

barplot(rev(variable_importance_info$PCR),horiz=T,xlim=c(-0.5,0.5), axes=F,col=rep(cm.colors(1),each=15), main='PCR')
text(seq(from=1.5,length.out=135,by=1.2),x=-0.1, label=rev(variable_importance_info$features))
axis(1,c(0,0.1,0.2,0.3,0.4),c(0,0.1,0.2,0.3,0.4))

barplot(rev(variable_importance_info$GL),horiz=T,xlim=c(-0.5,0.5), axes=F,col=rep(cm.colors(1),each=15), main='Genelized Linear')
text(seq(from=1.5,length.out=135,by=1.2),x=-0.1, label=rev(variable_importance_info$features))
axis(1,c(0,0.1,0.2,0.3,0.4, 0.5),c(0,0.1,0.2,0.3,0.4, 0.5))

barplot(rev(variable_importance_info$BoostTree),horiz=T,xlim=c(-0.5,0.5), axes=F,col=rep(cm.colors(1),each=15), main='Boost Tree')
text(seq(from=1.5,length.out=135,by=1.2),x=-0.1, label=rev(variable_importance_info$features))
axis(1,c(0,0.1,0.2,0.3,0.4),c(0,0.1,0.2,0.3,0.4))

