import torchviz, torch, torchinfo, sklearn.metrics
import tensorflow as tf
import pandas as pd
import numpy as np

def VisualizeModel(Model, Type, OutputFile, InputSize = None, Print = False, Device = None):
    if Type == "Tensorflow":
        tf.keras.utils.plot_model(Model, to_file = f"{OutputFile}", show_shapes=True, show_layer_names=True)
    elif Type == "Pytorch":
        if InputSize == None: raise KeyError("InputSize not exist!")
        elif Device == None: raise KeyError("Device not exist!")
        else:
            FakeInput = torch.zeros(InputSize, dtype=torch.float, requires_grad=False).to(Device)
            _ = torchviz.make_dot(Model(FakeInput), params=dict(list(Model.named_parameters()))).render(OutputFile, format="png")
    else:
        raise KeyError("Wrong Model Type!")
    if Print: print(f"Generate file {OutputFile}.")

def GetModelSummary(Model, Type, InputSize = None):
    if Type == "Tensorflow":
        return Model.summary()
    elif Type == "Pytorch":
            if InputSize == None: raise KeyError("InputSize not exist!")
            else: return torchinfo.summary(Model, input_size = InputSize)
    else:
        raise KeyError("Wrong Model Type!")

def SaveModel(Model, Type, OutputFile, Print = False):
    # OutputFileName without extension 
    if Type == "Tensorflow":
        OutputFile = f"{OutputFile}.h5"
        Model.save(OutputFile)
    elif Type == "Pytorch":
        OutputFile = f"{OutputFile}.pt"
        torch.save(Model, OutputFile)
    else: 
        raise KeyError("Wrong Model Type!")
    if Print: print(f"Model Saved to {OutputFile}.")

def LoadModel(InputFile, Type, Print = False):
    if Type == "Tensorflow":
        InputFile = f"{InputFile}.h5"
        Model = tf.keras.models.load_model(InputFile)
    elif Type == "Pytorch":
        InputFile = f"{InputFile}.pt"
        Model = torch.load(InputFile)
    else: 
        raise KeyError("Wrong Model Type!")
    if Print: print(f"Load Model from {InputFile}.")
    return Model

def EvaluateModel(Model, XData, yData, NumberofClass):
    # This also outputs probability of each class
    print("-------------Evaluate Results-------------")
    yPredictLabel = pd.DataFrame(np.argmax(Model.predict(XData), axis = 1), columns = ["PredictLabel"])
    yPredictArray = pd.DataFrame(Model.predict(XData), columns = ["PredictClass{}".format(i) for i in range(NumberofClass)])
    yPredict = pd.concat([yPredictLabel, yPredictArray], axis = 1)

    Acc = sum(yPredict.PredictLabel == yData) / len(yData)
    print("Accuracy: {}".format(Acc))

    F1 = sklearn.metrics.f1_score(yData, yPredict.PredictLabel, average = "weighted")
    print("F1 score: {}".format(F1))

    CofMat = sklearn.metrics.confusion_matrix(yData, yPredict.PredictLabel)
    CofMat = (CofMat.T / CofMat.sum(axis = 1)).T
    print(CofMat)
    print("------------------------------------------")
    return Acc, F1, yPredict, CofMat

def GetModelIntermediateOutput(Model, Type, Dataset, Indices, LayerNames, Device = None, InputSize = None):
    # Input Indices and LayerNames are lists
    # This returns a disctinary of LayerName as key
    def PytorchGetFeatures(LayerName):
        def hook(model, input, output):
            Features[LayerName] = output.detach()
        return hook

    Features = {}
    if Type == "Tensorflow":
        for LayerName in LayerNames:
            Features[LayerName] = tf.keras.models.Model(Model.input, Model.get_layer(LayerName).output)(Dataset[Indices])
    elif Type == "Pytorch":
        if Device == None: raise KeyError("Device not exist!")
        elif InputSize == None: raise KeyError("InputSize not exist!")
        else:
            SampleTensor, _ = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(Dataset, Indices), batch_size = len(Indices), num_workers = 0, shuffle = False)))
            SampleTensor = SampleTensor.view(InputSize).to(Device)
            for LayerName in LayerNames:
                getattr(Model, LayerName).register_forward_hook(PytorchGetFeatures(LayerName))
            _ = Model(SampleTensor)
            for LayerName in LayerNames:
                Features[LayerName] = Features[LayerName].cpu().numpy()
    else:
        raise KeyError("Wrong Model Type!")
    return Features
