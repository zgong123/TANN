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

from CommonNN import EvaluateModel, SaveModel
from datetime import datetime, timedelta

from TrainFunctions import GetHistData, GetFeatures, GetLabels, ProcessData, GetRowToUpload, ReadThresholds
from ModelFunctions import TrainModel
from CommonGZ import CheckExist, Tee, GetUTCNow, DateTimeRange, GetCommandLineArguments
from CommonGC import UploadRow

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

Thresholds = None

if OldThreshhold: ThresholdFile = "ThresholdOld.csv"
else: ThresholdFile = "Threshold.csv"

if FixedThreshold and OldThreshhold: KindIdentifier = f"{NumberOfClasses}Class_{NumberOfTrainDays}Day_OldThreshold"
elif FixedThreshold: KindIdentifier = f"{NumberOfClasses}Class_{NumberOfTrainDays}Day_FixedThreshold"
else: KindIdentifier = f"{NumberOfClasses}Class_{NumberOfTrainDays}Day"

for TargetDateTime in DateTimeRange(StartDate + timedelta(days = NumberOfTrainDays - 1), EndDate, timedelta(days = 1), True):

    TrainDays = [TargetDateTime - timedelta(days = D) for D in range(NumberOfTrainDays)]
    Year = str(TargetDateTime.year).zfill(4); Month = str(TargetDateTime.month).zfill(2); Day = str(TargetDateTime.day).zfill(2)

    ErrOutFolder = os.path.join(OutputRoot, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}", Symbol, "ErrOut")
    TrainErrFile = os.path.join(ErrOutFolder, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}_{Symbol}_{Year}.err")

    ModelOutFolder = os.path.join(OutputRoot, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}", Symbol, "ModelOut")

    _ = CheckExist([ErrOutFolder, ModelOutFolder], "Folder", True)

    SysStderr = sys.stderr
    PrintErr = open(TrainErrFile, "a+", buffering = 1)
    sys.stderr = Tee(SysStderr, PrintErr)

    print("-------------------------------")
    print("Symbol: {}".format(Symbol))
    print("Target Date: {}-{}-{}".format(Year, Month, Day))
    print("Train Day: ", end = "")
    for TrainDay in TrainDays: print(f"{TrainDay.year}-{TrainDay.month}-{TrainDay.day} , ", end = "") 
    print("")
    print("Prediction Horizon: {}".format(PredictionHorizon))
    print(f"Fixed Threshold: {FixedThreshold}")

    TimeDelta = timedelta(days = NumberOfTrainDays)
    AskM1, BidM1 = GetHistData(Symbol = Symbol, StartDateTime = TargetDateTime - timedelta(days = NumberOfTrainDays - 1), TimeDelta = TimeDelta, Type = "M1Correct", Folder = os.path.join(DataRoot, "HistData"))
    AskTicks, BidTicks = GetHistData(Symbol = Symbol,  StartDateTime = TargetDateTime - timedelta(days = NumberOfTrainDays - 1), TimeDelta = TimeDelta, Type = "TickNew", Folder = os.path.join(DataRoot, "HistData"))

    if (len(AskTicks) < max(WindowSizes)) or (len(BidTicks) < max(WindowSizes)): 
        print("Not Enough Features!")
        continue
    
    if FixedThreshold: Thresholds = ReadThresholds(os.path.join(DataRoot, ThresholdFile), Symbol, PredictionHorizon)
    print(Thresholds)

    Features = GetFeatures(AskTicks, BidTicks, FeatureLength = max(WindowSizes))
    Labels, Thresholds = GetLabels(BidM1[["DateTime", "BidClose"]], FeatureName = "BidClose", WindowSizes = [PredictionHorizon], NumberOfClasses = NumberOfClasses, Thresholds = Thresholds)

    XData, yData = ProcessData(Labels, Features, LabelName = "M{}Label".format(PredictionHorizon), FeatureNames = ["AskFeature", "BidFeature"])

    Model = TrainModel(XData, yData, ImageSize = XData.shape[1:], Verbose = 0)

    Acc, F1, yPredict, CofMat = EvaluateModel(Model, XData, yData, NumberOfClasses)

    if CofMat.shape != (NumberOfClasses, NumberOfClasses):
        print("Not enough number of classes!")
        continue

    if True: SaveModel(Model, "Tensorflow", os.path.join(ModelOutFolder, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}_{Symbol}_Train_{Year}{Month}{Day}"), True)

    Row = GetRowToUpload(Symbol, TargetDateTime, Scenario, PredictionHorizon, Thresholds, Acc, F1, CofMat)
    RowName = "TANN_{}".format(GetUTCNow()[0].strftime("%Y%m%d%H%M%S%f"))
    if True: UploadRow(Kind = f"TANN_{KindIdentifier}", Name = RowName, Row = Row)

    sys.stderr = SysStderr
    PrintErr.close()