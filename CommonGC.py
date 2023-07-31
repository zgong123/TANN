from google.cloud import datastore
import pandas as pd
import os, tqdm

def UploadRow(Kind, Name, Row, CredentialsFile = "GoogleCloud.json", Project = "phd-project-1-243716", Print = True):

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CredentialsFile

    # Instantiates a client
    DatastoreClient = datastore.Client(project = Project)
    KeyToUpload = DatastoreClient.key(Kind, Name)

    # Prepares the new entity
    EntityToUpload = datastore.Entity(key = KeyToUpload)

    for k,v in Row.items(): EntityToUpload[k] = v

    # Saves the entity
    DatastoreClient.put(EntityToUpload)

    if Print: print('Row Saved to Datastore: {} ---- {}'.format(EntityToUpload.key.kind, EntityToUpload.key.name))

    return None

def QueryRow(Kind, Queries, CredentialsFile = "GoogleCloud.json", Project = "phd-project-1-243716", Print = False):
    Output= []
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CredentialsFile
    Client = datastore.Client(project = Project)
    Query = Client.query(kind = Kind)

    for Q in Queries: Query = Query.add_filter(Q[0], Q[1], Q[2])

    Result = list(Query.fetch())
    if Print: print(f"Get {len(Result)} Results!")

    for Row in Result:
        RowDict = dict(Row)
        RowDict.update({"Kind": Row.key.kind, "Name": Row.key.name})
        Output.append(RowDict)
    return Output

def QueryToDataFrame(QueryResults, Columns, Separator = None, SaveFile = None, DropNa = False, SplitModel = False, ModelTypes = ["LDA", "KN", "SVC", "RF", "LR", "TANN"]):
    Output = {}
    Rawdf = pd.DataFrame(QueryResults)[Columns]
    if "DateTime" in Columns: Rawdf["DateTime"] = Rawdf["DateTime"].dt.tz_localize(None)
    if DropNa: 
        Rawdf = Rawdf.dropna(axis = 0, how = "any")
        Rawdf = Rawdf[Rawdf.TestAcc != 0]
    Rawdf = Rawdf.rename(columns = {"TrainAcc": "TANNTrainAcc", "TrainF1": "TANNTrainF1", "TestAcc": "TANNTestAcc", "TestF1": "TANNTestF1", "TestClass1": "TANNTestClass1", "TestClass2": "TANNTestClass2", "TestClass3": "TANNTestClass3"})
    if SplitModel:
        print("Split Model...")
        df = pd.DataFrame(columns = ["DateTime", "Symbol", "PredictionHorizon", "ModelType", "TrainAcc", "TrainF1", "TestAcc", "TestF1"])
        for _, r in tqdm.tqdm(Rawdf.iterrows(), total = Rawdf.shape[0]):
            for m in ModelTypes:
                df = df.append({"DateTime": r.DateTime, "Symbol": r.Symbol, "PredictionHorizon": r.PredictionHorizon, "Threshold1": r.Threshold1, "Threshold2": r.Threshold2,
                "ModelType": m, 
                "TrainAcc": r[f"{m}TrainAcc"], "TrainF1": r[f"{m}TrainF1"], "TestAcc": r[f"{m}TestAcc"], "TestF1": r[f"{m}TestF1"]}, ignore_index = True)
    else: df = Rawdf.copy()
    if Separator:
        SeparatorList = list(set(df[Separator]))
        for Se in SeparatorList: 
            Output[Se] = df[df[Separator] == Se].reset_index(drop = True)
        if SaveFile:
            Writer = pd.ExcelWriter(SaveFile)
            for Se in SeparatorList: 
                print(f"Write Category: {Se}, length: {len(Output[Se])}")
                Output[Se].to_excel(Writer, sheet_name = Se, index = False)
            Writer.save()
            print(f"Result saved to {SaveFile}.")
    else:
        Output["DataFrame"] = df
        if SaveFile:
            Writer = pd.ExcelWriter(SaveFile)
            Output["DataFrame"].to_excel(Writer, index = False)
            Writer.save()
            print(f"Result saved to {SaveFile}.")
    return Output