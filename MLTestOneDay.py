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

from pandas.core.arrays import boolean
sys.path.insert(0, UtilitiesRoot)
sys.path.insert(0, ProjectRoot)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from datetime import datetime, timedelta
import sklearn.metrics
from joblib import load
import os

from TrainFunctions import GetHistData, GetFeatures, GetLabels, ProcessData, ReadThresholds
from CommonGZ import GetBusinessDay, GetCommandLineArguments, DateTimeRange, CheckExist
from CommonGC import UploadRow, QueryRow

RequiredArguments = {"Symbol": "Symbol of Currency",
                    "StartDate": "Start date",
                    "EndDate": "End date",
                    "ModelType": "Type of Model",
                    "PredictionHorizon": "Prediction Horizon",
                    "FixedThreshold": "Use Fixed Threshold"}

Arguments = GetCommandLineArguments(RequiredArguments)

Symbol = Arguments["Symbol"]
StartDate = datetime.strptime(Arguments["StartDate"], "%Y%m%d")
EndDate = datetime.strptime(Arguments["EndDate"], "%Y%m%d")

ModelType = Arguments["ModelType"]
PredictionHorizon = int(Arguments["PredictionHorizon"])
FixedThreshold = bool(int(Arguments["FixedThreshold"]))

NumberOfClasses = 3
Scenario = "S1"
WindowSizes = [60, 100, 140, 180, 220, 260]
Indicators = ["RSI", "WMA", "SMA", "CMO", "ROC"]


Thresholds = None

if FixedThreshold: KindIdentifier = f"{NumberOfClasses}Class_1Day_FixedThreshold"
else: KindIdentifier = f"{NumberOfClasses}Class_1Day"

for TargetDateTime in DateTimeRange(StartDate, EndDate, timedelta(days = 1), True):
    ModelDateTime = GetBusinessDay(TargetDateTime, -1)[0]

    Year = str(TargetDateTime.year).zfill(4); Month = str(TargetDateTime.month).zfill(2); Day = str(TargetDateTime.day).zfill(2)
    ModelYear = str(ModelDateTime.year).zfill(4); ModelMonth = str(ModelDateTime.month).zfill(2); ModelDay = str(ModelDateTime.day).zfill(2)

    ModelInputFolder = os.path.join(OutputRoot, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}", Symbol, f"{ModelType}ModelOut")
    ModelInputFile = os.path.join(ModelInputFolder, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}_{Symbol}_Train_{ModelYear}{ModelMonth}{ModelDay}.joblib")

    print("Symbol: {}".format(Symbol))
    print("TestDate: {}-{}-{}".format(Year, Month, Day))
    print("TrainDate: {}-{}-{}".format(ModelYear, ModelMonth, ModelDay))
    print("Prediction Horizon: {}".format(PredictionHorizon))
    print("Model Type: {}".format(ModelType))
    print("Fixed Threshold: {}".format(FixedThreshold))

    if not CheckExist([ModelInputFile], "File", False):
        print("Model Not Exists!")
        continue

    TimeDelta = timedelta(days = 1)
    AskM1, BidM1 = GetHistData(Symbol = Symbol, StartDateTime = TargetDateTime, TimeDelta = TimeDelta, Type = "M1Correct", Folder = os.path.join(DataRoot, "HistData"))
    AskTicks, BidTicks = GetHistData(Symbol = Symbol,  StartDateTime = TargetDateTime, TimeDelta = TimeDelta, Type = "TickNew", Folder = os.path.join(DataRoot, "HistData"))
    if (len(AskTicks) < max(WindowSizes)) or (len(BidTicks) < max(WindowSizes)): 
        print("Not Enough Features!")
        continue

    if FixedThreshold: Thresholds = ReadThresholds(os.path.join(DataRoot, "Threshold.csv"), Symbol, PredictionHorizon)
    
    Features = GetFeatures(AskTicks, BidTicks)
    Labels, Thresholds = GetLabels(BidM1[["DateTime", "BidClose"]], FeatureName = "BidClose", WindowSizes = [PredictionHorizon], NumberOfClasses = NumberOfClasses, Thresholds = Thresholds)
    print(Thresholds)

    XData, yData = ProcessData(Labels, Features, LabelName = "M{}Label".format(PredictionHorizon), FeatureNames = ["AskFeature", "BidFeature"], Type = "ML")

    Model = load(ModelInputFile)

    yPredict = Model.predict(XData)
    TestAcc = sum(yPredict == yData)/len(yData)
    TestF1 = sklearn.metrics.f1_score(yData, yPredict, average="weighted")
    print(f"{ModelType} TestAcc: {TestAcc}")
    print(f"{ModelType} TestF1: {TestF1}")

    if True:
        QueryOutput = QueryRow(Kind = f"TANN_{KindIdentifier}", Queries = [("DateTime", "=", TargetDateTime), ("Symbol", "=", Symbol), ("PredictionHorizon", "=", PredictionHorizon)], Print = False)
        if len(QueryOutput) > 0:
          Row = QueryOutput[0]
          RowKind = Row["Kind"]; RowName = Row["Name"]; RowValue = Row
          del RowValue["Kind"]; del RowValue["Name"]
          RowValue[f"{ModelType}TestAcc"] = float(TestAcc); RowValue[f"{ModelType}TestF1"] = float(TestF1)
          UploadRow(Kind = RowKind, Name = RowName, Row = RowValue)
        else:
          print("No QueryOutput!")
    print("------------------")
