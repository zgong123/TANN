# -*- coding: utf-8 -*-
ProjectRoot = "/mnt/projects/TANN/"
UtilitiesRoot = "/mnt/projects/Utilities"

import sys
sys.path.insert(0, UtilitiesRoot)
sys.path.insert(0, ProjectRoot)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from CommonNN import EvaluateModel, SaveModel
from datetime import datetime, timedelta
from numpy import concatenate
import os

from TrainFunctions import GetHistData, GetFeatures, GetLabels, ProcessData
from ModelFunctions import TrainModel
from CommonGZ import CheckExist, Tee, GetCommandLineArguments, DateTimeRange

RequiredArguments = {"StartDate": "Start date",
                    "EndDate": "End date"}
Arguments = GetCommandLineArguments(RequiredArguments)

StartDate = datetime.strptime(Arguments["StartDate"], "%Y%m%d")
EndDate = datetime.strptime(Arguments["EndDate"], "%Y%m%d")

Symbols = ["USDNOK", "AUDJPY", "GBPJPY", "EURSEK", "EURUSD", "EURTRY", "USDCHF", "GBPCAD", "USDDKK", "USDJPY"]

NumberOfClasses = 3
PredictionHorizon = 15
Scenario = "S1"
WindowSizes = [60, 100, 140, 180, 220, 260]
Indicators = ["RSI", "WMA", "SMA", "CMO", "ROC"]

for TargetDateTime in DateTimeRange(StartDate, EndDate, timedelta(days = 1), True):
  Year = str(TargetDateTime.year).zfill(4); Month = str(TargetDateTime.month).zfill(2); Day = str(TargetDateTime.day).zfill(2)

  ErrOutFolder = os.path.join(ProjectRoot, "Output", f"{NumberOfClasses}Class_1Day_M{PredictionHorizon}_{Scenario}", "Universal", "ErrOut")
  TrainErrFile = os.path.join(ErrOutFolder, f"{NumberOfClasses}Class_1Day_M{PredictionHorizon}_{Scenario}_Universal_{Year}.err")

  ModelOutFolder = os.path.join(ProjectRoot, "Output", f"{NumberOfClasses}Class_1Day_M{PredictionHorizon}_{Scenario}", "Universal", "ModelOut")

  _ = CheckExist([ErrOutFolder, ModelOutFolder], "Folder", True)

  SysStderr = sys.stderr
  PrintErr = open(TrainErrFile, "a+", buffering = 1)
  sys.stderr = Tee(SysStderr, PrintErr)

  print("Train Universial")
  print("TrainDate: {}-{}-{}".format(Year, Month, Day))
  print("Prediction Horizon: {}".format(PredictionHorizon))

  XDataAll = []; yDataAll = []
  print("Read Data ", end = "")
  for Symbol in Symbols:
    TimeDelta = timedelta(days = 1)
    AskM1, BidM1 = GetHistData(Symbol = Symbol, StartDateTime = TargetDateTime, TimeDelta = TimeDelta, Type = "M1Correct", Folder = os.path.join(ProjectRoot, "HistData"))
    AskTicks, BidTicks = GetHistData(Symbol = Symbol,  StartDateTime = TargetDateTime, TimeDelta = TimeDelta, Type = "TickNew", Folder = os.path.join(ProjectRoot, "HistData"))
    if (len(AskTicks) < max(WindowSizes)) or (len(BidTicks) < max(WindowSizes)): 
      print(f"{Symbol} Not Enough Features!", end="")
      continue
    Features = GetFeatures(AskTicks, BidTicks)
    Labels, Thresholds = GetLabels(BidM1[["DateTime", "BidClose"]], FeatureName = "BidClose", WindowSizes = [PredictionHorizon], NumberOfClasses = NumberOfClasses)
    XData, yData = ProcessData(Labels, Features, LabelName = "M{}Label".format(PredictionHorizon), FeatureNames = ["AskFeature", "BidFeature"], Print = False)
    XDataAll.append(XData); yDataAll.append(yData)
    print(".", end = "")
  print("Finish")

  XDataAll = concatenate(XDataAll, axis=0)
  yDataAll = concatenate(yDataAll, axis=0)
  print(f"XData Shape: {XDataAll.shape}")
  print(f"yData Shape: {yDataAll.shape}")

  Model = TrainModel(XDataAll, yDataAll, ImageSize = XData.shape[1:], Verbose = 0, NumberOfEpochs = 200)

  Acc, F1, yPredict, CofMat = EvaluateModel(Model, XData, yData, NumberOfClasses)
  if True: SaveModel(Model, "Tensorflow", os.path.join(ModelOutFolder, f"{NumberOfClasses}Class_1Day_M{PredictionHorizon}_{Scenario}_Universal_Train_{Year}{Month}{Day}"), True)

sys.stderr = SysStderr
PrintErr.close()