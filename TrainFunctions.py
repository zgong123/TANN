
from Utilities import DownloadData, FeatureToImages
from datetime import datetime, timedelta
from os.path import join, exists
from re import sub
from pandas import read_pickle, DataFrame, merge, read_csv
from numpy import array, stack, zeros
from CommonGZ import CheckExist, DateTimeRange, RoundDateTime

def GetRowToUpload(Symbol, DateTime, Scenario, PredictionHorizon, Thresholds, TrainAcc, TrainF1, TrainCofMat):
    Row = {}
    Row["Symbol"] = str(Symbol)
    Row["DateTime"] = DateTime
    Row["DayOfWeek"] = int(DateTime.weekday()) #Monday is 0 and Sunday is 6.
    Row["Scenario"] = str(Scenario)
    Row["PredictionHorizon"] = int(PredictionHorizon)
    Row["Threshold1"] = float(Thresholds[f"M{PredictionHorizon}"][0])
    Row["Threshold2"] = float(Thresholds[f"M{PredictionHorizon}"][1])
    Row["TrainAcc"] = float(TrainAcc)
    Row["TrainF1"] = float(TrainF1)
    Row["TrainClass1"] = float(TrainCofMat[0,0])
    Row["TrainClass2"] = float(TrainCofMat[1,1])
    Row["TrainClass3"] = float(TrainCofMat[2,2])
    return Row

def ReadThresholds(ThresholdFile, Symbol, PredictionHorizon):
    Threshold = read_csv(ThresholdFile) 
    Out = {"Symbol": Symbol, f"M{PredictionHorizon}": [Threshold.loc[Threshold.Symbol == Symbol, f"M{PredictionHorizon}Down"].values[0], Threshold.loc[Threshold.Symbol == Symbol, f"M{PredictionHorizon}Up"].values[0]]}
    return Out

def GetThresholds(Labels, WindowSizes = [2, 5, 10, 15], NumberOfClasses = 3):
    LabelNames = ["M{}".format(WindowSize) for WindowSize in WindowSizes]
    Thresholds = {LabelName: None for LabelName in LabelNames}
    Precentiles = [round( i / NumberOfClasses, 2) for i in range(1, NumberOfClasses)]
    for LabelName in LabelNames:
        Stat = Labels[LabelName].describe(percentiles = Precentiles)
        Thresholds[LabelName] = [Stat["{0:.0%}".format(Precentile)] for Precentile in Precentiles]

    return Thresholds

def GetHistData(Symbol, StartDateTime, TimeDelta, Type = "Tick", Folder = "/content/drive/MyDrive/Data/HistData"):
    Year = format(int(StartDateTime.year), '04'); Month = format(int(StartDateTime.month), '02')

    Folder = join(Folder, Type, Symbol)
    if not CheckExist([Folder], Type = "Folder", Create = False):
        raise ValueError(f"{Folder} not exist!")

    EndDateTime = StartDateTime + TimeDelta
    BidData, AskData = None, None

    if Type is "M1Correct":
        ReadFile = join(Folder, "DAT_ASCII_{}_M1_{}.xz".format(Symbol, Year))
        Data = read_pickle(ReadFile)   #Bid Only
        BidData = Data[(Data.DateTime >= StartDateTime) & (Data.DateTime <= EndDateTime)].reset_index()
    if (Type is "Tick") or (Type is "TickNew"):
        ReadFile = join(Folder, Year, "DAT_ASCII_{}_T_{}{}.xz".format(Symbol, Year, Month))
        Data = read_pickle(ReadFile) 
        AskData = Data[(Data.DateTime >= StartDateTime) & (Data.DateTime <= EndDateTime)][["DateTime", "Ask"]].rename(columns={"Ask": "AskPrice"}).reset_index()
        BidData = Data[(Data.DateTime >= StartDateTime) & (Data.DateTime <= EndDateTime)][["DateTime", "Bid"]].rename(columns={"Bid": "BidPrice"}).reset_index()
    return AskData, BidData

def GetWholeDayTicks(Fx = None, Symbol = "EUR/USD", Year = 2020, Month = 12, Day = 30, Folder = "/content/drive/My Drive/Data/FXCMDemoStream/", UseTqdm = True):

    SavingFolder = join(Folder, "{}".format(sub(r"[^\w]", "", Symbol)), "{}_{}{}{}".format(sub(r"[^\w]", "", Symbol), str(Year).zfill(4), str(Month).zfill(2), str(Day).zfill(2)))
    CheckExist([SavingFolder], Type = "Folder", Create = True)
    SavingFileName = "{}_{}{}{}".format(sub(r"[^\w]", "", Symbol), str(Year).zfill(4), str(Month).zfill(2), str(Day).zfill(2))

    if exists(join(SavingFolder, "{}_{}.xz".format(SavingFileName, "BidTicks"))) and exists(join(SavingFolder, "{}_{}.xz".format(SavingFileName, "AskTicks"))):

        print("File {} exists. Read Tick Files.".format(SavingFileName))
        AskTicksFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "AskTicks"))
        BidTicksFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "BidTicks"))
        AskTicks = read_pickle(AskTicksFile)
        BidTicks = read_pickle(BidTicksFile)

    else:
        print("Download tick data from FXCM server.")

        AskTicks, BidTicks = DownloadData(Fx = Fx, Symbol = Symbol, Frequency = "t1", StartTime = datetime(Year, Month, Day, 0, 0, 0), EndTime = datetime(Year, Month, Day, 23, 59, 59), UseTqdm = UseTqdm)

        AskTicksFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "AskTicks"))
        BidTicksFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "BidTicks"))
        BidTicks.to_pickle(BidTicksFile)
        AskTicks.to_pickle(AskTicksFile)

    return AskTicks, BidTicks, AskTicksFile, BidTicksFile

def GetWholeDayMinutes(Fx = None, Symbol = "EUR/USD", Year = 2020, Month = 12, Day = 30, Folder = "/content/drive/My Drive/Data/FXCMDemoStream/", UseTqdm = True):

    SavingFolder = join(Folder, "{}".format(sub(r"[^\w]", "", Symbol)), "{}_{}{}{}".format(sub(r"[^\w]", "", Symbol), str(Year).zfill(4), str(Month).zfill(2), str(Day).zfill(2)))
    CheckExist([SavingFolder], Type = "Folder", Create = True)
    SavingFileName = "{}_{}{}{}".format(sub(r"[^\w]", "", Symbol), str(Year).zfill(4), str(Month).zfill(2), str(Day).zfill(2))

    if exists(join(SavingFolder, "{}_{}.xz".format(SavingFileName, "BidMinutes"))) and exists(join(SavingFolder, "{}_{}.xz".format(SavingFileName, "AskMinutes"))):

        print("File {} exists. Read Minute Files.".format(SavingFileName))
        AskMinutesFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "AskMinutes"))
        BidMinutesFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "BidMinutes"))
        AskMinutes = read_pickle(AskMinutesFile)
        BidMinutes = read_pickle(BidMinutesFile)

    else:
        print("Download minute data from FXCM server.")

        AskMinutes, BidMinutes = DownloadData(Fx = Fx, Symbol = Symbol, Frequency = "m1", StartTime = datetime(Year, Month, Day, 0, 0, 0), EndTime = datetime(Year, Month, Day, 23, 59, 59), UseTqdm = UseTqdm)

        AskMinutesFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "AskMinutes"))
        BidMinutesFile = join(SavingFolder, "{}_{}.xz".format(SavingFileName, "BidMinutes"))
        BidMinutes.to_pickle(BidMinutesFile)
        AskMinutes.to_pickle(AskMinutesFile)
        print("Ask and Bid minutes saved to {}.".format(SavingFileName))

    return AskMinutes, BidMinutes, AskMinutesFile, BidMinutesFile

def GetLabels(M1Prices, FeatureName = "MidClose", WindowSizes = [2, 5, 10, 15], NumberOfClasses = 3, Thresholds = None, UseTqdm = False):
    MaxWindowSize = max(WindowSizes)
    StartDateTime = M1Prices.at[MaxWindowSize, "DateTime"]
    EndDateTime = M1Prices.at[len(M1Prices) - MaxWindowSize - 1, "DateTime"]

    LabelNames = ["M{}".format(WindowSize) for WindowSize in WindowSizes]
    Labels = DataFrame(columns=["DateTime"] + LabelNames)
    for TargetDateTime in DateTimeRange(StartDateTime, EndDateTime, timedelta(minutes = 1), UseTqdm = UseTqdm):
        Row = {ColName: None for ColName in list(Labels.columns)}
        Row["DateTime"] = TargetDateTime
        for WindowSize in WindowSizes:
            AverageBefore = M1Prices[(M1Prices.DateTime >= TargetDateTime - timedelta(minutes = WindowSize)) & (M1Prices.DateTime <  TargetDateTime)][FeatureName].mean()
            AverageAfter =  M1Prices[(M1Prices.DateTime > TargetDateTime) & (M1Prices.DateTime <=  TargetDateTime + timedelta(minutes = WindowSize))][FeatureName].mean()
            Row["M{}".format(WindowSize)] = (AverageAfter - AverageBefore) / AverageBefore * 10000

        Labels = Labels.append(Row, ignore_index = True)

    if Thresholds is None: Thresholds = GetThresholds(Labels, WindowSizes = WindowSizes, NumberOfClasses = NumberOfClasses)

    for LabelName in LabelNames:
        Labels["{}Label".format(LabelName)] = NumberOfClasses - 1
        for i in range(NumberOfClasses - 1):
            if i == 0:
                Labels.at[Labels[LabelName] < Thresholds[LabelName][0], "{}Label".format(LabelName)] = 0 
            else:
                Labels.at[(Labels[LabelName] >= Thresholds[LabelName][i - 1]) & (Labels[LabelName] < Thresholds[LabelName][i]), "{}Label".format(LabelName)] = i

    return Labels, Thresholds

def GetLabelsFast(M1Prices, FeatureName = "MidClose", WindowSizes = [2, 5, 10, 15], NumberOfClasses = 3, Thresholds = None):
    LabelNames = ["M{}".format(WindowSize) for WindowSize in WindowSizes]
    Labels = DataFrame(columns=["DateTime"] + LabelNames)
    for WindowSize in WindowSizes:
        M1Prices[f"MAAfterM{WindowSize}"] = M1Prices[FeatureName].rolling(window = WindowSize).mean().shift(-WindowSize)
        M1Prices[f"MABeforeM{WindowSize}"] = M1Prices[FeatureName].rolling(window = WindowSize).mean().shift(1)
        M1Prices[f"M{WindowSize}"] = (M1Prices[f"MAAfterM{WindowSize}"] - M1Prices[f"MABeforeM{WindowSize}"]) / M1Prices[f"MABeforeM{WindowSize}"]  * 10000

    M1Prices = M1Prices.dropna(axis = 0).reset_index()
    Labels = M1Prices[["DateTime"] + LabelNames].copy()

    if Thresholds is None: Thresholds = GetThresholds(Labels, WindowSizes = WindowSizes, NumberOfClasses = NumberOfClasses)

    for LabelName in LabelNames:
        Labels.loc[:, "{}Label".format(LabelName)] = NumberOfClasses - 1
        for i in range(NumberOfClasses - 1):
            if i == 0:
                Labels.loc[Labels[LabelName] < Thresholds[LabelName][0], "{}Label".format(LabelName)] = 0 
            else:
                Labels.loc[(Labels[LabelName] >= Thresholds[LabelName][i - 1]) & (Labels[LabelName] < Thresholds[LabelName][i]), "{}Label".format(LabelName)] = i

    return Labels, Thresholds

def GetFeatures(AskTicks, BidTicks, FeatureLength = 260, MaxWindowSize = 15):
    StartDateTime = RoundDateTime(AskTicks.at[FeatureLength, "DateTime"], "Minute") + timedelta(minutes = 1)
    EndDateTime = RoundDateTime(AskTicks.at[len(AskTicks) - 1, "DateTime"], "Minute") + timedelta(minutes = 1)
    Features = DataFrame(columns=["DateTime", "AskFeature", "BidFeature"])
    for TargetDateTime in DateTimeRange(StartDateTime, EndDateTime, timedelta(minutes = 1)):
        Row = {ColName: None for ColName in list(Features.columns)}
        Row["DateTime"] = TargetDateTime
        Row["AskFeature"] = array(AskTicks[AskTicks.DateTime < TargetDateTime][-FeatureLength:].AskPrice)
        Row["BidFeature"] = array(BidTicks[BidTicks.DateTime < TargetDateTime][-FeatureLength:].BidPrice)
        Features = Features.append(Row, ignore_index = True)

    return Features

def ProcessData(Labels, Features, LabelName = "M15Label", FeatureNames = ["AskFeature", "BidFeature"], Type = "NN", Print = True):
    if Type is "NN":
        Combined = merge(left = Labels, right = Features, on = ["DateTime"], how = "inner")[[LabelName] + FeatureNames].copy()
        XData = stack(list(map(FeatureToImages, [Combined[FeatureName] for FeatureName in FeatureNames])), axis = 3).astype('float32')
        yData = array(Combined[LabelName]).astype('float32')
    elif Type is "ML":
        Combined = merge(left = Labels, right = Features, on = ["DateTime"], how = "inner")[[LabelName, FeatureNames[0]]].copy()
        XData = array(list(map(array, Combined[FeatureNames[0]]))).astype('float32')
        yData = array(Combined[LabelName]).astype('float32')
    else:
        raise KeyError("Wrong Model Type!")
    if Print:
        print("XData shape: {}".format(XData.shape))
        print("yData shape: {}".format(yData.shape))
    return XData, yData