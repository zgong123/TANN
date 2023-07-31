# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
ProjectRoot = "/mnt/Projects/TANN/"
UtilitiesRoot = "/mnt/Projects/Utilities"
DataRoot = "/mnt/ProjectData/TANN"
OutputRoot = "/mnt/ProjectData/TANN/Output"

#ProjectRoot = "X:\\TANN"
#UtilitiesRoot = "X:\\Utilities"
#DataRoot = "Z:\\TANN"
#OutputRoot = "Z:\\TANN\\Output\\Trail"

import sys
sys.path.insert(0, UtilitiesRoot)
sys.path.insert(0, ProjectRoot)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from CommonNN import EvaluateModel, LoadModel
from datetime import datetime, timedelta

from TrainFunctions import GetHistData, GetFeatures, GetLabels, ProcessData, ReadThresholds
from CommonGZ import GetBusinessDay, GetCommandLineArguments, DateTimeRange, CheckExist
from CommonGC import UploadRow, QueryRow

NumberOfClasses = 3

RequiredArguments = {"Symbol": "Symbol of Currency",
                    "StartDate": "Start date",
                    "EndDate": "End date",
                    "PredictionHorizon": "Prediction Horizon",
                    "FixedThreshold": "Use Fixed Threshold",
                    "NumberOfTrainDays": "Number of Days to Train",
                    "OldThreshold": "Use Old Threshold",
                    "Scenario": "Scenario Name"}

Arguments = GetCommandLineArguments(RequiredArguments)

Symbol = str(Arguments["Symbol"])
StartDate = datetime.strptime(Arguments["StartDate"], "%Y%m%d")
EndDate = datetime.strptime(Arguments["EndDate"], "%Y%m%d")
PredictionHorizon = int(Arguments["PredictionHorizon"])
FixedThreshold = bool(int(Arguments["FixedThreshold"]))
NumberOfTrainDays = int(Arguments["NumberOfTrainDays"])
OldThreshold = bool(int(Arguments["OldThreshold"]))
Scenario = str(Arguments["Scenario"])

if Scenario == "S1": 
    WindowSizes = [60, 100, 140, 180, 220, 260]
    Indicators = ["RSI", "WMA", "SMA", "CMO", "ROC"]
elif Scenario == "S2":
    WindowSizes = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    Indicators = ["RSI", "WMA", "SMA", "CMO", "ROC"]
else:
    raise ValueError("Wrong Scenario Name!")
    
if OldThreshhold: ThresholdFile = "ThresholdOld.csv"
else: ThresholdFile = "Threshold.csv"

if FixedThreshold and OldThreshhold: KindIdentifier = f"{NumberOfClasses}Class_{NumberOfTrainDays}Day_OldThreshold"
elif FixedThreshold: KindIdentifier = f"{NumberOfClasses}Class_{NumberOfTrainDays}Day_FixedThreshold"
else: KindIdentifier = f"{NumberOfClasses}Class_{NumberOfTrainDays}Day"

Thresholds = None
TrainDays = 1  #Get the model from one day before

for TargetDateTime in DateTimeRange(StartDate, EndDate, timedelta(days = 1), True):
    ModelDateTime = GetBusinessDay(TargetDateTime, -TrainDays)[0]

    Year = str(TargetDateTime.year).zfill(4); Month = str(TargetDateTime.month).zfill(2); Day = str(TargetDateTime.day).zfill(2)
    ModelYear = str(ModelDateTime.year).zfill(4); ModelMonth = str(ModelDateTime.month).zfill(2); ModelDay = str(ModelDateTime.day).zfill(2)

    ModelInputFolder = os.path.join(OutputRoot, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}", Symbol, "ModelOut")
    ModelInputFile = os.path.join(ModelInputFolder, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}_{Symbol}_Train_{ModelYear}{ModelMonth}{ModelDay}")

    print("Kind: {}".format(KindIdentifier))
    print("Symbol: {}".format(Symbol))
    print("Test Date: {}-{}-{}".format(Year, Month, Day))
    print("Train Date: {}-{}-{}".format(ModelYear, ModelMonth, ModelDay))
    print("Prediction Horizon: {}".format(PredictionHorizon))
    print("Fixed Threshold: {}".format(FixedThreshold))

    if not CheckExist([f"{ModelInputFile}.h5"], "File", False):
        print("Model Not Exists!")
        continue

    TimeDelta = timedelta(days = 1)
    AskM1, BidM1 = GetHistData(Symbol = Symbol, StartDateTime = TargetDateTime, TimeDelta = TimeDelta, Type = "M1Correct", Folder = os.path.join(DataRoot, "HistData"))
    AskTicks, BidTicks = GetHistData(Symbol = Symbol,  StartDateTime = TargetDateTime, TimeDelta = TimeDelta, Type = "TickNew", Folder = os.path.join(DataRoot, "HistData"))
    if (len(AskTicks) < max(WindowSizes)) or (len(BidTicks) < max(WindowSizes)): 
        print("Not Enough Features!")
        continue

    if FixedThreshold: Thresholds = ReadThresholds(os.path.join(DataRoot, "Threshold.csv"), Symbol, PredictionHorizon)

    Features = GetFeatures(AskTicks, BidTicks, FeatureLength = max(WindowSizes))
    Labels, Thresholds = GetLabels(BidM1[["DateTime", "BidClose"]], FeatureName = "BidClose", WindowSizes = [PredictionHorizon], NumberOfClasses = NumberOfClasses, Thresholds = Thresholds)
    print(Thresholds)
    XData, yData = ProcessData(Labels, Features, LabelName = "M{}Label".format(PredictionHorizon), FeatureNames = ["AskFeature", "BidFeature"])

    Model = LoadModel(ModelInputFile, "Tensorflow", False)

    Acc, F1, yPredict, CofMat = EvaluateModel(Model, XData, yData, NumberOfClasses)

    if True:
        QueryOutput = QueryRow(Kind = f"TANN_{KindIdentifier}", Queries = [("DateTime", "=", TargetDateTime), ("Symbol", "=", Symbol), ("PredictionHorizon" ,"=", PredictionHorizon), ("Scenario", "=", Scenario)], Print = False)
        if len(QueryOutput) > 0:
            Row = QueryOutput[0]
            RowKind = Row["Kind"]; RowName = Row["Name"]; RowValue = Row
            del RowValue["Kind"]; del RowValue["Name"]
            RowValue["TestAcc"] = float(Acc); RowValue["TestF1"] = float(F1)
            RowValue["TestClass1"] = float(CofMat[0,0]); RowValue["TestClass2"] = float(CofMat[1,1]); RowValue["TestClass3"] = float(CofMat[2,2])
            UploadRow(Kind = RowKind, Name = RowName, Row = RowValue)
        else:
          print("No QueryOutput!")
    print("------------------")