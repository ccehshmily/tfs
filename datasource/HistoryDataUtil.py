# -*- coding: utf-8 -*-
import io

class HistoryDataUtil:
    """
    Util for processing history data crawled from yahoo finance api.

    The data is get from the following api:
    http://ichart.yahoo.com/table.csv?s=GOOG&a=01&b=01&c=2012&d=31&e=12&f=2015&g=d
    The data get from this api is in the following format:
    Date,Open,High,Low,Close,Volume,Adj Close
    2016-05-24,706.859985,720.969971,706.859985,720.090027,1920400,720.090027
    2016-05-23,706.530029,711.478027,704.179993,704.23999,1320900,704.23999
    ......

    This util is going to convert the raw data into (train, validation)
    data which tensorflow can use.

    The (features, metric) is (all info for past N days, price change after L days)
    """

    def genFeatures(dailyData):
        featureList = []
        for data in dailyData:
            dataFloat = [float(oneData) for oneData in data]
            dataModified = dataFloat[:-2] + [dataFloat[-1]/1000, dataFloat[-1]]
            featureList = featureList + dataModified[3:]
        return featureList

    def genLabel(dataNow, dataLater):
        priceNow = float(dataNow[-1])
        priceLater = float(dataLater[-1])
        pChangePercent = (priceLater - priceNow) / priceNow

        labelList = [0] * 2
        if pChangePercent <= 0:
            labelList[0] = 1
        else:
            labelList[1] = 1

        return labelList

    @staticmethod
    def extractDailyData(filename):
        """
        Convert raw data to daily data, represented as list of lists, ordered by
        date ascending.
        """
        dailyData = []
        dataFile = open(filename, 'r')
        dataLines = dataFile.readlines()
        for dataLine in dataLines[1:]:
            dataBlocks = dataLine.split(',')
            dataToday = dataBlocks[0].split('-') + dataBlocks[1:]
            dailyData.append(dataToday)
        dailyData.reverse()
        return dailyData

    @staticmethod
    def generateTFData(dailyData, N=10, L=1, featureGenerator=genFeatures, labelGenerator=genLabel):
        """
        Convert daily data to format that TF can use. Divided to 0.8 training
        and 0.2 testing/validation.
        """
        nDays = len(dailyData) - L - N + 1
        nDaysTrain = int(nDays * 0.8)
        nDaysTest = nDays - nDaysTrain

        trainFeature = []
        trainLabel = []
        for i in range(nDaysTrain):
            trainFeature.append(featureGenerator(dailyData[i:i+N]))
            trainLabel.append(labelGenerator(dailyData[i+N-1], dailyData[i+N+L-1]))
        trainData = (trainFeature, trainLabel)

        testFeature = []
        testLabel = []
        for i in range(nDaysTest):
            testFeature.append(featureGenerator(dailyData[i+nDaysTrain:i+nDaysTrain+N]))
            testLabel.append(labelGenerator(dailyData[i+N+nDaysTrain-1], dailyData[i+N+L+nDaysTrain-1]))
        testData = (testFeature, testLabel)

        return (trainData, testData)
