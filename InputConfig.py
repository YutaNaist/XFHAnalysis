import json

# import os


class InputConfig:
    jsonDict = {}
    Directory = ""
    FileNameOriginal = ""

    def __init__(self, DirName, FileName):
        self.FileNameOriginal = FileName
        self.jsonDict = json.load(open(DirName + FileName, "r"))
        self.Directory = DirName
        # self.Directory = os.path.dirname(FileName) + "/"

    def saveJson(self, filename="InputConfigTemp.json"):
        outputFileName = self.Directory + filename
        with open(outputFileName, "w") as file:
            json.dump(self.jsonDict, file, indent=2)
        return outputFileName

    def saveJsonOriginal(self):
        outputFileName = self.Directory + self.FileNameOriginal
        with open(outputFileName, "w") as file:
            json.dump(self.jsonDict, file, indent=2)
        return outputFileName

    def updateDirectory(self, Directory):
        self.jsonDict["InputDirectory"] = Directory
        self.jsonDict["OutputDirectory"] = Directory
        return self.saveJson()

    def updateValue(self, Key, Value):
        self.jsonDict[Key] = Value

    def updateValues(self, Keys, Values):
        for i, Key in enumerate(Keys):
            self.jsonDict[Key] = Values[i]

    def updateValuesByDictionary(self, Dictionary):
        Keys = Dictionary.keys()
        Values = Dictionary.values()
        for i, Key in enumerate(Keys):
            self.jsonDict[Key] = Values[i]

    def getValue(self, Key):
        return self.jsonDict[Key]
