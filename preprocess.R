
# Read data from git
library(RCurl)
original <- getURL("https://raw.githubusercontent.com/anshengli2/cs6375/test/nyse.csv")
original <- read.csv(text = original)
data_to_convert <- original

names(original) # get columns name
str(original) # get the data type

# convert char to numeric by columns order
library(stringr) 
# Sales
Value_unity <- ifelse(str_detect(data_to_convert$Sales, 'M'), 1e6, ifelse(str_detect(data_to_convert$Sales, 'B'), 1e9, 1))
data_to_convert$Sales<- Value_unity * as.numeric(str_remove(data_to_convert$Sales, 'B|M'))

# Income
Value_unity <- ifelse(str_detect(data_to_convert$Income, 'M'), 1e6, ifelse(str_detect(data_to_convert$Income, 'B'), 1e9, 1))
data_to_convert$Income <- Value_unity * as.numeric(str_remove(data_to_convert$Income, 'B|M'))

data_to_convert$Analyst.Rating..1.buy.5.sell. <- as.numeric(data_to_convert$Analyst.Rating..1.buy.5.sell.)
data_to_convert$price.to.earnings <- as.numeric(data_to_convert$price.to.earnings)
data_to_convert$Price.Earnings.to.Growth <- as.numeric(data_to_convert$Price.Earnings.to.Growth)
data_to_convert$Price.to.Sales.Ratio <- as.numeric(data_to_convert$Price.to.Sales.Ratio)
data_to_convert$Price.to.Book <- as.numeric(data_to_convert$Price.to.Book)
data_to_convert$Price.to.free.cash.flow <- as.numeric(data_to_convert$Price.to.free.cash.flow)
data_to_convert$debt.to.equity <- as.numeric(data_to_convert$debt.to.equity)
data_to_convert$EPS..ttm. <- as.numeric(data_to_convert$EPS..ttm.)

# Inst.Own
data_to_convert$Inst.Own <- 0.01 * as.numeric(str_remove(data_to_convert$Inst.Own, '%'))
data_to_convert$ROE <- 0.01 * as.numeric(str_remove(data_to_convert$ROE, '%'))
data_to_convert$ROI <- 0.01 * as.numeric(str_remove(data_to_convert$ROI, '%'))
data_to_convert$Profit.Margin <- 0.01 * as.numeric(str_remove(data_to_convert$Profit.Margin, '%'))

# Shs.Outstand
Value_unity <- ifelse(str_detect(data_to_convert$Shs.Outstand, 'M'), 1e6, ifelse(str_detect(data_to_convert$Shs.Outstand, 'B'), 1e9, 1))
data_to_convert$Shs.Outstand <- Value_unity * as.numeric(str_remove(data_to_convert$Shs.Outstand, 'B|M'))

data_to_convert$Beta <- as.numeric(data_to_convert$Beta)

str(data_to_convert) # get the data type after conversion

# rename the data and save the data to specified path
nyse <- data_to_convert
write.csv(nyse,"C:/Users/tony5/Desktop/CS_6375/Project/nyse_convert.csv", row.names = FALSE)