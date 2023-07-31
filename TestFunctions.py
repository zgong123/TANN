from Utilities import DownloadData
from datetime import datetime, timedelta
from numpy import array, mean, argmax
from re import sub
from google.cloud import datastore

def GetLiveTick(Fx = None, Symbol = "EUR/USD", Year = 2020, Month = 12, Day = 30, Hour = 12, Minute = 10, Length = 260):
    TargetTime = datetime(Year, Month, Day, Hour, Minute)

    print("Download tick data from FXCM server.")
    LookBackMinutes = 2
    while True:
        print("Trail {}".format(int(LookBackMinutes/2)))
        AskTicks, BidTicks = DownloadData(Fx = Fx, Symbol = Symbol, Frequency = "t1", StartTime = TargetTime - timedelta(minutes = LookBackMinutes), EndTime = TargetTime, ShowTimeSpent = False)
        if len(AskTicks) >= Length: break
        else: LookBackMinutes = LookBackMinutes + 2

    AskTicks = array(AskTicks.AskPrice)[-Length:]
    BidTicks = array(BidTicks.BidPrice)[-Length:]
    return AskTicks, BidTicks

def GetLiveMinute(Fx = None, Symbol = "EUR/USD", Year = 2020, Month = 12, Day = 30, Hour = 12, Minute = 10, Length = 15*2):
    TargetTime = datetime(Year, Month, Day, Hour, Minute)

    print("Download minute data from FXCM server.")
    AskMinutes, BidMinutes = DownloadData(Fx = Fx, Symbol = Symbol, Frequency = "m1", StartTime = TargetTime - timedelta(minutes = Length), EndTime = TargetTime, ShowTimeSpent = False)

    return AskMinutes, BidMinutes

def GetTrueLabel(AskMinutes, BidMinutes, Thresholds, PredictionHorizon = 15):
    LabelName = "M{}".format(PredictionHorizon)
    MidClose = mean([array(AskMinutes.AskClose), array(BidMinutes.BidClose)], axis = 0)
    PctChange = (mean(MidClose[-PredictionHorizon:], axis = 0) - mean(MidClose[:PredictionHorizon], axis = 0)) / mean(MidClose[:PredictionHorizon], axis = 0) * 10000

    Label = len(Thresholds[LabelName])
    for i in range(len(Thresholds[LabelName])):

        if PctChange < Thresholds[LabelName][i]:
             Label = i
             break

    return AskMinutes.DateTime.iloc[PredictionHorizon], Label, PctChange

def GetPredictionLabel(Pred, Threshold = 0.34):
    Labels = []
    for i in range(len(Pred)):
        if Pred[i][argmax(Pred[i])] > Threshold: Labels.append(argmax(Pred[i]))
        else: Labels.append(1)
    return array(Labels)


def UploadLabel(Symbol, LabelTime, PctChange, Label, Kind = "FXCMLive", PrintOut = True):

    Symbol = sub(r"[^\w]", "", Symbol)

    #Values = {"Year": int(LabelTime.year), "Month": int(LabelTime.month), "Day":  int(LabelTime.day), "Hour": int(LabelTime.hour), "Minute": int(LabelTime.minute)}

    # Instantiates a client
    DatastoreClient = datastore.Client(project = "phd-project-1-243716")

    # The kind for the new entity
    #Kind = Values.get("Kind")
    # The name/ID for the new entity
    #Name = Values.get("Name")
    # The Cloud Datastore key for the new entity
    Name = "{}_{}_{}_{}_{}_{}".format(Symbol, str(LabelTime.year).zfill(4), str(LabelTime.month).zfill(2), str(LabelTime.day).zfill(2), str(LabelTime.hour).zfill(2), str(LabelTime.minute).zfill(2))
    Query = DatastoreClient.query(kind = Kind)
    Query.add_filter("__key__", "=", DatastoreClient.key(Kind, Name))
    
    Values = {}
    for Element in Query.fetch():
      for Key in Element:
        Values[Key] = Element[Key]

    Values["Label"] = Label
    try:
        Values["Accuracy"] = float(1) if Label == Values["Predict"] else float(0)
    except KeyError:
        pass
    Values["PctChange"] = float(PctChange)

    KeyToUpload = DatastoreClient.key(Kind, Name)
    # Prepares the new entity
    EntityToUpload = datastore.Entity(key = KeyToUpload)
    
    for k,v in Values.items(): EntityToUpload[k] = v

    # Saves the entity
    DatastoreClient.put(EntityToUpload)

    if PrintOut: print('Label and Accuracy Saved to Datastore: {} ---- {}'.format(EntityToUpload.key.kind, EntityToUpload.key.name))

    return None