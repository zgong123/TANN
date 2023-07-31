# -*- coding: utf-8 -*-
#ProjectRoot = "/mnt/Projects/TANN/"
#UtilitiesRoot = "/mnt/Projects/Utilities"
#DataRoot = "/mnt/ProjectData/TANN"

ProjectRoot = "X:\\TANN"
UtilitiesRoot = "X:\\Utilities"
DataRoot = "Z:\\TANN"
OutputRoot = "Z:\\TANN\\Output\\Trail"

import sys

from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import date_range
sys.path.insert(0, UtilitiesRoot)
sys.path.insert(0, ProjectRoot)

from datetime import datetime, timedelta

from TrainFunctions import GetHistData, GetLabelsFast
from CommonGZ import CheckExist, Tee
import os

Symbols = ["AUDJPY", "EURSEK", "EURTRY", "EURUSD", "GBPCAD", "GBPJPY", "USDCHF", "USDDKK", "USDJPY", "USDNOK"]
#Symbols = ["USDDKK"]#, "USDJPY", "USDNOK"]
PredictionHorizons = [5, 10, 15]

Threshold = DataFrame(columns = ["Symbol"] + [f"M{PredictionHorizon}Threshold" for PredictionHorizon in PredictionHorizons])
Years = ["2014", "2015", "2016", "2017", "2018"]

ThresholdDataFrame = DataFrame(columns= ["Symbol"] + [f"M{PredictionHorizon}" for PredictionHorizon in PredictionHorizons])
for Symbol in Symbols:
    OneSymbolData = DataFrame(columns=["DateTime", "BidClose"])
    print(f"{Symbol}, ", end = "")
    for Year in Years:
        print(f"{Year}, ", end = "")
        StartDateTime = datetime(int(Year), 1, 1); EndDateTime = datetime(int(Year), 12, 31)
        _, BidM1 = GetHistData(Symbol = Symbol, StartDateTime = StartDateTime, TimeDelta = EndDateTime - StartDateTime, Type = "M1Correct", Folder = os.path.join(DataRoot, "HistData"))
        OneSymbolData = OneSymbolData.append(BidM1[["DateTime", "BidClose"]], ignore_index = True)
    print("")
    _, Threshold = GetLabelsFast(OneSymbolData, FeatureName = "BidClose", WindowSizes = PredictionHorizons, NumberOfClasses = 3)
    Threshold["Symbol"] = Symbol
    print(Threshold)
    ThresholdDataFrame = ThresholdDataFrame.append(Threshold, ignore_index = True)

ThresholdDataFrame.to_csv(os.path.join(OutputRoot, f"Threshold.csv"))