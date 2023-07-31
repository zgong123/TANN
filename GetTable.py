
#ProjectRoot = "/mnt/projects/TANN/"
#UtilitiesRoot = "/mnt/projects/Utilities"

#ProjectRoot = "X:\\TANN"
#UtilitiesRoot = "X:\\Utilities"

ProjectRoot = "G:\\My Drive\\_myProject\\TANN"
UtilitiesRoot = "G:\\My Drive\\_myProject\\Utilities"

import sys
sys.path.insert(0, UtilitiesRoot)
sys.path.insert(0, ProjectRoot)

from CommonGC import QueryRow, QueryToDataFrame
from datetime import datetime

StartDateTime = datetime(2014,1,1)
EndDateTime = datetime(2018,12,31)

ModelTypes = ["TANN", "LDA", "SVC", "LR", "RF", "KN"]
#ModelTypes = ["TANN"]

#Kind = "TANN_3Class_1Day_FixedThresholds"
#Kind = "TANN_3Class_1Day_OldThreshold"
#Kind = "TANN_3Class_1Day"
#Kind = "TANN_3Class_2Day"
#Kind = "TANN_3Class_3Day"
#Kind = "TANN_3Class_1Day_Universal"
Kind = "TANN_3Class_1Day_500Epoch_Universal"

Queries = [("DateTime",">=", StartDateTime), ("DateTime", "<", EndDateTime)]
#Queries = [("DateTime",">=", StartDateTime), ("DateTime", "<", EndDateTime), ("PredictionHorizon", "=", 5)]
#Queries = [ ("PredictionHorizon", "=", 5)]

#Columns = ["DateTime", "PredictionHorizon", "Symbol", "TrainAcc", "LDATrainAcc", "SVCTrainAcc", "KNTrainAcc", "RFTrainAcc", "LRTrainAcc", "TestAcc", "LDATestAcc", "SVCTestAcc", "KNTestAcc", "RFTestAcc", "LRTestAcc", "TrainF1", "LDATrainF1", "SVCTrainF1", "KNTrainF1", "RFTrainF1", "LRTrainF1", "TestF1", "LDATestF1", "SVCTestF1", "KNTestF1", "RFTestF1", "LRTestF1", "Threshold1", "Threshold2",  "TestClass1", "TestClass2", "TestClass3"]
Columns = ["DateTime", "PredictionHorizon", "Symbol","Scenario", "TrainAcc", "TestAcc", "TrainF1", "TestF1", "Threshold1", "Threshold2", "TestClass1", "TestClass2", "TestClass3"]

print(f"Kind: {Kind}")
print("Start Date: {}, End Date: {}".format(StartDateTime, EndDateTime))
print("Querying...", end = "")

Row = QueryRow(Kind, Queries, CredentialsFile = "D:\\CloudCredentials\\GoogleCloud.json")

print(f"Total Number of Rows: {len(Row)}")
_ = QueryToDataFrame(Row, Columns, None, SaveFile = f"{Kind}.xlsx", DropNa = True, SplitModel = True, ModelTypes = ModelTypes)

