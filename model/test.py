from ..datasource.HistoryDataUtil import HistoryDataUtil as dUtil

dailyData = dUtil.extractDailyData('tfs/datasource/sampledata/sampleGOOG.txt')
dataSet = dUtil.generateTFData(dailyData)

trainData = dataSet[0]
trainLabels = trainData[1]

labelCount = [0] * 10
for label in trainLabels:
    for i in range(10):
        if label[i] == 1:
            labelCount[i] = labelCount[i] + 1
            break

print labelCount
