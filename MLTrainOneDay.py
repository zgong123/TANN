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

from datetime import datetime, timedelta

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import sklearn.metrics
from joblib import dump
import os

from TrainFunctions import GetHistData, GetFeatures, GetLabels, ProcessData, ReadThresholds
from CommonGZ import CheckExist, Tee, GetCommandLineArguments, DateTimeRange
from CommonGC import UploadRow, QueryRow

RequiredArguments = {"Symbol": "Symbol of Currency",
                    "StartDate": "Start date",
                    "EndDate": "End date",
                    "ModelType": "Type of Model",
                    "PredictionHorizon": "Prediction Horizon",
                    "FixedThreshold": "Use Fixed Threshold"}
                    
Arguments = GetCommandLineArguments(RequiredArguments)

Symbol = str(Arguments["Symbol"])
StartDate = datetime.strptime(Arguments["StartDate"], "%Y%m%d")
EndDate = datetime.strptime(Arguments["EndDate"], "%Y%m%d")

ModelType = str(Arguments["ModelType"])
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
  Year = str(TargetDateTime.year).zfill(4); Month = str(TargetDateTime.month).zfill(2); Day = str(TargetDateTime.day).zfill(2)

  ErrOutFolder = os.path.join(OutputRoot, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}", Symbol, "ErrOut")
  TrainErrFile = os.path.join(ErrOutFolder, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}_{Symbol}_{Year}_{ModelType}.err")

  ModelOutFolder = os.path.join(OutputRoot, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}", Symbol, f"{ModelType}ModelOut")

  _ = CheckExist([ErrOutFolder, ModelOutFolder], "Folder", True)

  SysStderr = sys.stderr
  PrintErr = open(TrainErrFile, "a+", buffering = 1)
  sys.stderr = Tee(SysStderr, PrintErr)

  print("Symbol :{}".format(Symbol))
  print("TrainDate: {}-{}-{}".format(Year, Month, Day))
  print("Prediction Horizon: {}".format(PredictionHorizon))
  print("Model Type: {}".format(ModelType))
  print("Fixed Threshold: {}".format(FixedThreshold))

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

  if ModelType == "SVC":
    Model = SVC()
  elif ModelType == "LDA":
    Model = LinearDiscriminantAnalysis(solver = "svd",  store_covariance = True, tol = 0.01)
  elif ModelType == "RF":
    Model = RandomForestClassifier()
  elif ModelType == "KN":
    Model = KNeighborsClassifier()
  elif ModelType == "LR":
    Model = LogisticRegression()

  Model.fit(XData, yData)

  yPredict = Model.predict(XData)
  TrainAcc = sum(yPredict == yData)/len(yData)
  TrainF1 = sklearn.metrics.f1_score(yData, yPredict, average="weighted")
  print(f"{ModelType} TrainAcc: {TrainAcc}")
  print(f"{ModelType} TrainF1: {TrainF1}")

  if True: dump(Model,  os.path.join(ModelOutFolder, f"{KindIdentifier}_M{PredictionHorizon}_{Scenario}_{Symbol}_Train_{Year}{Month}{Day}.joblib"))

  if True:
    QueryOutput = QueryRow(Kind = f"TANN_{KindIdentifier}", Queries = [("DateTime", "=", TargetDateTime), ("Symbol", "=", Symbol), ("PredictionHorizon", "=", PredictionHorizon)], Print = False)
    if len(QueryOutput) > 0:
      Row = QueryOutput[0]
      RowKind = Row["Kind"]; RowName = Row["Name"]; RowValue = Row
      del RowValue["Kind"]; del RowValue["Name"]
      RowValue[f"{ModelType}TrainAcc"] = float(TrainAcc); RowValue[f"{ModelType}TrainF1"] = float(TrainF1)
      UploadRow(Kind = RowKind, Name = RowName, Row = RowValue)
    else:
      print("No QueryOutput!")
  print("------------------")

sys.stderr = SysStderr
PrintErr.close()
