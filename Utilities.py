from collections import defaultdict
from time import time
from forexconnect import fxcorepy, ForexConnect
from tqdm import trange
from pytz import utc
from re import sub
from pathlib import Path
from datetime import datetime, timedelta 
from pandas import DataFrame, to_datetime, read_csv
from numpy import array, vectorize, stack, flip, mean
from talib import RSI, WMA, EMA, SMA, CMO, ROC, MACD


import os
import json


def _getRSI(Vector):
  return RSI(Vector, timeperiod = len(Vector) - 1)[-1]

def _getWMA(Vector):
  return WMA(Vector, timeperiod = len(Vector) - 1)[-2]
'''          
def _getEMA(Vector):
  return EMA(Vector, timeperiod = len(Vector) - 1)[-2]
'''        
def _getSMA(Vector):
  return SMA(Vector, timeperiod = len(Vector) - 1)[-2]

def _getCMO(Vector):
  return CMO(Vector, timeperiod = len(Vector) - 1)[-1]
          
def _getROC(Vector):
  return ROC(Vector, timeperiod = len(Vector) - 1)[-1]

def _getMACD(Vector):
  return MACD(Vector)[2].diff().iloc[-1]

def ProcessRow(row, Frequency):
    if Frequency == "t1":
        BidRow = {"DateTime": to_datetime(str(row['Date'])), "BidPrice": row["Bid"]}
        AskRow = {"DateTime": to_datetime(str(row['Date'])), "AskPrice": row["Ask"]}
    else:
        BidRow = {"DateTime": to_datetime(str(row['Date'])),
            "BidOpen": row["BidOpen"],
            "BidHigh": row["BidHigh"],
            "BidLow": row["BidLow"],
            "BidClose": row["BidClose"],
            "BidVolume": row["Volume"]}

        AskRow = {"DateTime": to_datetime(str(row['Date'])),
            "AskOpen": row["AskOpen"],
            "AskHigh": row["AskHigh"],
            "AskLow": row["AskLow"],
            "AskClose": row["AskClose"],
            "AskVolume": row["Volume"]}
    
    return BidRow, AskRow


def VectorToImages(Vector, WindowSizes = [60, 100, 140, 180, 220, 260], Indicators = ["RSI", "WMA", "SMA", "CMO", "ROC"]):
    Image = DataFrame(index = WindowSizes, columns = Indicators)
    Vector = flip(Vector)
    for Indicator in Indicators:
        for WindowSize in WindowSizes:
          Image.at[WindowSize, Indicator] = eval("_get{}(Vector[:(WindowSize + 1)])".format(Indicator))
    
    return array(Image)

def FeatureToImages(Features):
    return stack(vectorize(VectorToImages)(array(Features)))

def SessionStatusChanged(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))

def DownloadData(Fx = None, Symbol = "EUR/USD", Frequency = "m1", 
          StartTime = datetime.now(utc) - timedelta(minutes = 15), EndTime = datetime.now(utc), 
          UseTqdm = False, ShowTimeSpent = True):
    NeedLogout = False
    FXCMCredentials = json.load(open(os.environ['FOREXCONNECT_CREDENTIALS']))
    if Fx is None:
        Fx = ForexConnect()
        Fx.login(FXCMCredentials["username"], FXCMCredentials["password"], FXCMCredentials["url"], FXCMCredentials["connection"], None, None, SessionStatusChanged)
        NeedLogout = True
    
    BidPrices = DataFrame()
    AskPrices = DataFrame()

    t0 = time()
 
    History = Fx.get_history(Symbol, Frequency, StartTime, EndTime)

    if NeedLogout:
      try: Fx.logout()
      except Exception as e: print("Exception: " + str(e))

    print("Processing {} rows of {} data.".format(len(History), Frequency))

    if UseTqdm:
        for i in trange(len(History)):    
            BidRow, AskRow = ProcessRow(History[i], Frequency)
            BidPrices = BidPrices.append(BidRow, ignore_index = True)
            AskPrices = AskPrices.append(AskRow, ignore_index = True)
    else:
        for i in range(len(History)):    
            BidRow, AskRow = ProcessRow(History[i], Frequency)
            BidPrices = BidPrices.append(BidRow, ignore_index = True)
            AskPrices = AskPrices.append(AskRow, ignore_index = True)

    if ShowTimeSpent: print("*{}".format(GetRunningTime(t0 = t0, t1 = time())))

    return AskPrices, BidPrices 

def GetMidMinutes(AskMinutes, BidMinutes, OutputFile = None):

    if not BidMinutes.DateTime.equals(AskMinutes.DateTime):
        print("DateTime of Bid and Ask tables are NOT identical!")
        return None
    else:
        MidMinutes = DataFrame(columns = ["DateTime", "MidClose", "MidHigh", "MidLow", "MidOpen"])
        MidMinutes.DateTime = AskMinutes.DateTime
        MidMinutes.MidClose = mean([array(AskMinutes.AskClose), array(BidMinutes.BidClose)], axis = 0)
        MidMinutes.MidHigh = mean([array(AskMinutes.AskHigh), array(BidMinutes.BidHigh)], axis = 0)
        MidMinutes.MidLow = mean([array(AskMinutes.AskLow), array(BidMinutes.BidLow)], axis = 0)
        MidMinutes.MidOpen = mean([array(AskMinutes.AskOpen), array(BidMinutes.BidOpen)], axis = 0)
        if OutputFile is not None:
            MidMinutes.to_pickle(OutputFile, protocol = 4)
            print("MidMinutes Data Saved to: {}".format(OutputFile))
        return MidMinutes

def GetAccountBalance(Fx = None, FileName = "AccountBalance.csv"):  
  NeedLogout = False
  FXCMCredentials = json.load(open(os.environ['FOREXCONNECT_CREDENTIALS']))
  if Fx is None:
      Fx = ForexConnect()
      Fx.login(FXCMCredentials["username"], FXCMCredentials["password"], FXCMCredentials["url"], FXCMCredentials["connection"], None, None, None)
      NeedLogout = True

  Account = Fx.table_manager.get_table(ForexConnect.ACCOUNTS)[0]
  AccountID = Account.account_id
  Balance = Account.balance
  UTCNow, Year, Month, Day, _, _, _ = GetUTCNow()

  if NeedLogout:
    try: Fx.logout()
    except Exception as e: print("Exception: " + str(e))
  
  UTCNow = UTCNow.strftime("%Y-%m-%d %H:%M:%S")
  
  if Path(FileName).exists():
    BalanceTable = read_csv(FileName)
    BalanceTable = BalanceTable.append({"DateTime": UTCNow, "AccountID": AccountID, "Balance": Balance, "Change": 0}, ignore_index = True)
    BalanceTable["Change"] = BalanceTable["Balance"].diff()
    Change = BalanceTable.Change.iloc[-1]
    BalanceTable.to_csv(FileName, index = False)
  else:
    BalanceTable = DataFrame(columns = ["DateTime", "AccountID", "Balance", "Change"])
    BalanceTable = BalanceTable.append({"DateTime": UTCNow, "AccountID": AccountID, "Balance": Balance, "Change": 0}, ignore_index = True)
    Change = "0"
    BalanceTable.to_csv(FileName, index = False)

  return "Date: {}-{}-{}, Balance: {}, Change: {:.2f}".format(Year, Month, Day, Balance, Change), str(BalanceTable)

