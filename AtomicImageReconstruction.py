# %%
import numpy as np
import json
import datetime
import os
# from numba import jit


def readHologramFromCSV(fileList):
    sizeFileList = len(fileList)
    Data = []
    for iFile in range(sizeFileList):
        Data.append(np.loadtxt(fileList[iFile], delimiter=","))
    Data = np.array(Data)
    return Data


def writeIgorText(array, fileName, minX, stepX, minY, stepY, minZ, stepZ):
    sizeX = array.shape[0]
    sizeY = array.shape[1]
    sizeZ = array.shape[2]
    with open(fileName, mode='w') as file:
        file.write("IGOR\r")
        file.write("WAVES/D/N=({0},{1},{2})	noname\r".format(
            sizeX, sizeY, sizeZ))
        file.write("BEGIN\r")
        for z in range(sizeZ):
            for x in range(sizeX):
                stringArray = ""
                for y in range(sizeY):
                    stringArray += "\t" + str(array[x][y][z])
                stringArray += "\r"
                file.write(stringArray)
        file.write("END\r")
        file.write("X SetScale/P x {0},{1},"
                   ",noname; SetScale/P y {2},{3},"
                   ",noname; SetScale/P z {4},{5},"
                   ",noname\r".format(minX, stepX, minY, stepY, minZ, stepZ))


def AtomicImageReconstruction(inputJsonFile,
                              outputDirectory="",
                              isCalculateImage=True,
                              isInverseRealPart=False,
                              Comment=""):
    inputJson = json.load(open(inputJsonFile, 'r'))

    workingDir = inputJson["InputDirectory"]
    CurrentTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if outputDirectory == "":
        outputDirectory = workingDir + "AtomicImageReconstruction" + CurrentTime + "/"
        os.mkdir(outputDirectory)
    else:
        outputDirectory = inputJson["OutputDirectory"]

    jsonDict = {}
    jsonDict["Program"] = "AtomicImageReconstruction.py"
    jsonDict.update(inputJson)

    fileList = []
    energyList = []
    Files = inputJson["InputHologram"]
    for contents in Files.values():
        fileList.append(workingDir + contents["FileName"])
        energyList.append(contents["Energy"])
    # print(fileList)
    print(energyList)

    Hologram = readHologramFromCSV(fileList)
    print("Finish Reading File")

    minX = inputJson["MinX"]
    minY = inputJson["MinY"]
    minZ = inputJson["MinZ"]
    maxX = inputJson["MaxX"]
    maxY = inputJson["MaxY"]
    maxZ = inputJson["MaxZ"]
    stepX = inputJson["StepX"]
    stepY = inputJson["StepY"]
    stepZ = inputJson["StepZ"]

    startTheta = inputJson["StartTheta"]
    endTheta = inputJson["EndTheta"]
    stepTheta = inputJson["StepTheta"]
    startPhi = inputJson["StartPhi"]
    endPhi = inputJson["EndPhi"]
    stepPhi = inputJson["StepPhi"]

    mode = inputJson["Mode"]
    PhaseShift = inputJson["PhaseShift"] * 2 * np.pi

    sizeTheta = int((endTheta - startTheta) / stepTheta + 1)
    sizePhi = int((endPhi - startPhi) / stepPhi)
    sizeX = int((maxX - minX) / stepX + 1)
    sizeY = int((maxY - minY) / stepX + 1)
    sizeZ = int((maxZ - minZ) / stepX + 1)
    sizeEnergy = len(energyList)

    ThetaList = np.arange(startTheta, endTheta + stepTheta, stepTheta)
    PhiList = np.arange(startPhi, endPhi, stepTheta)

    CoordinateVector = np.empty((sizeX, sizeY, sizeZ, 4))
    # DistanceVector = np.empty((sizeX, sizeY, sizeZ))
    for iZ in range(sizeZ):
        z = minZ + stepZ * iZ
        for iY in range(sizeY):
            y = minY + stepY * iY
            for iX in range(sizeX):
                x = minX + stepX * iX
                Coordinate = np.array([x, y, z])
                # DistanceVector[iX][iY][iZ] = np.linalg.norm(Coordinate, ord=2)
                r = np.linalg.norm(Coordinate, ord=2)
                CoordinateVector[iX][iY][iZ] = np.append(Coordinate, r)
    print("Finish Making Coordinate Vector")

    # set k vector k = [kx, ky, kz, -|k||]
    CosTheta = np.cos(np.deg2rad(ThetaList))
    SinTheta = np.sin(np.deg2rad(ThetaList))
    SinPhi = np.sin(np.deg2rad(PhiList))
    CosPhi = np.cos(np.deg2rad(PhiList))
    kVector = np.empty((sizeEnergy, sizeTheta, sizePhi, 4))
    for iEnergy in range(sizeEnergy):
        k = 0.505 * energyList[iEnergy] / 1000
        for iTheta in range(sizeTheta):
            for iPhi in range(sizePhi):
                if mode == "Normal":
                    kVector[iEnergy][iTheta][iPhi] = np.array([
                        -SinTheta[iTheta] * CosPhi[iPhi],
                        -SinTheta[iTheta] * SinPhi[iPhi], -CosTheta[iTheta], 1
                    ])
                else:
                    kVector[iEnergy][iTheta][iPhi] = np.array([
                        SinTheta[iTheta] * CosPhi[iPhi],
                        SinTheta[iTheta] * SinPhi[iPhi], CosTheta[iTheta], 1
                    ])
        kVector[iEnergy] *= k
    kVector *= -1
    print("Finish Making Wave Number Vector")

    HoloSin = np.zeros_like(Hologram)
    for iTheta in range(sizeTheta):
        HoloSin[:, iTheta, :] = SinTheta[iTheta]
    kVector *= -1
    Hologram = Hologram * HoloSin
    AtomicImageReal = np.zeros((sizeX, sizeY, sizeZ))
    AtomicImageImag = np.zeros((sizeX, sizeY, sizeZ))
    for iZ in range(sizeZ):
        Z = minZ + stepZ * iZ
        for iX in range(sizeX):
            X = minX + stepX * iX
            display = "Z : {0:.3}, X : {1:.3}".format(Z, X)
            print("\r", display, end="")
            for iY in range(sizeY):
                if CoordinateVector[iX, iY, iZ, 3] > 1:
                    Phase = (np.tensordot(CoordinateVector[iX, iY, iZ, :],
                                          kVector,
                                          axes=((0), (3)))) + PhaseShift
                    AtomicImageReal[iX, iY,
                                    iZ] = np.tensordot(Hologram,
                                                       np.cos(Phase),
                                                       axes=((0, 1, 2), (0, 1,
                                                                         2)))
                    if isCalculateImage is True:
                        AtomicImageImag[iX, iY,
                                        iZ] = np.tensordot(Hologram,
                                                           np.sin(Phase),
                                                           axes=((0, 1, 2),
                                                                 (0, 1, 2)))
    print()
    fileNameReal = inputJson["InputAtomicImage"]["FileNameReal"]
    fileNameImag = inputJson["InputAtomicImage"]["FileNameImaginal"]
    if isInverseRealPart is True:
        writeIgorText(-1 * AtomicImageReal, outputDirectory + fileNameReal,
                      minX, stepX, minY, stepY, minZ, stepZ)
    else:
        writeIgorText(AtomicImageReal, outputDirectory + fileNameReal, minX,
                      stepX, minY, stepY, minZ, stepZ)
    if isCalculateImage is True:
        writeIgorText(AtomicImageImag, outputDirectory + fileNameImag, minX,
                      stepX, minY, stepY, minZ, stepZ)
    Files = {"RealPart": fileNameReal, "ImaginalPart": fileNameImag}
    jsonDict["OutputFile"] = Files
    jsonDict["Comment"] = Comment
    # with open(outputDirectory + "Result" + Comment + ".json", 'w') as file:
    #     json.dump(jsonDict, file, indent=2)
    logdict = {
        "program": "AtomicImageReconstruction",
        "outputdirectory": outputDirectory,
        "time": CurrentTime,
        "comment": Comment
    }
    logDirectory = inputJson["LogDirectory"]
    with open(logDirectory + "XFHCalculationLog.txt", 'a') as file:
        json.dump(logdict, file, indent=2)
    return outputDirectory

def AtomicImageReconstructionWithNumba(inputJsonFile,
                                       outputDirectory="",
                                       isCalculateImage=True,
                                       Comment=""):
    from numba import jit
    inputJson = json.load(open(inputJsonFile, 'r'))

    workingDir = inputJson["InputDirectory"]
    CurrentTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if outputDirectory == "":
        outputDirectory = workingDir + "AtomicImageReconstruction" + CurrentTime + "/"
        os.mkdir(outputDirectory)

    jsonDict = {}
    jsonDict["Program"] = "AtomicImageReconstruction.py"
    jsonDict.update(inputJson)

    fileList = []
    energyList = []
    Files = inputJson["InputHologram"]

    for contents in Files.values():
        fileList.append(workingDir + contents["FileName"])
        energyList.append(contents["Energy"])
    # print(fileList)
    print(energyList)

    Hologram = readHologramFromCSV(fileList)
    print("Finish Reading File")

    minX = inputJson["MinX"]
    minY = inputJson["MinY"]
    minZ = inputJson["MinZ"]
    maxX = inputJson["MaxX"]
    maxY = inputJson["MaxY"]
    maxZ = inputJson["MaxZ"]
    stepX = inputJson["StepX"]
    stepY = inputJson["StepY"]
    stepZ = inputJson["StepZ"]

    startTheta = inputJson["StartTheta"]
    endTheta = inputJson["EndTheta"]
    stepTheta = inputJson["StepTheta"]
    startPhi = inputJson["StartPhi"]
    endPhi = inputJson["EndPhi"]
    stepPhi = inputJson["StepPhi"]

    mode = inputJson["Mode"]
    PhaseShift = inputJson["PhaseShift"] * 2 * np.pi

    sizeTheta = int((endTheta - startTheta) / stepTheta + 1)
    sizePhi = int((endPhi - startPhi) / stepPhi)
    sizeX = int((maxX - minX) / stepX + 1)
    sizeY = int((maxY - minY) / stepX + 1)
    sizeZ = int((maxZ - minZ) / stepX + 1)
    sizeEnergy = len(energyList)

    ThetaList = np.arange(startTheta, endTheta + stepTheta, stepTheta)
    PhiList = np.arange(startPhi, endPhi, stepTheta)

    CoordinateVector = np.empty((sizeX, sizeY, sizeZ, 4))
    # DistanceVector = np.empty((sizeX, sizeY, sizeZ))
    for iZ in range(sizeZ):
        z = minZ + stepZ * iZ
        for iY in range(sizeY):
            y = minY + stepY * iY
            for iX in range(sizeX):
                x = minX + stepX * iX
                Coordinate = np.array([x, y, z])
                # DistanceVector[iX][iY][iZ] = np.linalg.norm(Coordinate, ord=2)
                r = np.linalg.norm(Coordinate, ord=2)
                CoordinateVector[iX][iY][iZ] = np.append(Coordinate, r)
    print("Finish Making Coordinate Vector")

    # set k vector k = [kx, ky, kz, -|k||]
    CosTheta = np.cos(np.deg2rad(ThetaList))
    SinTheta = np.sin(np.deg2rad(ThetaList))
    SinPhi = np.sin(np.deg2rad(PhiList))
    CosPhi = np.cos(np.deg2rad(PhiList))
    kVector = np.empty((sizeEnergy, sizeTheta, sizePhi, 4))
    for iEnergy in range(sizeEnergy):
        k = 0.505 * energyList[iEnergy] / 1000
        for iTheta in range(sizeTheta):
            for iPhi in range(sizePhi):
                if mode == "Normal":
                    kVector[iEnergy][iTheta][iPhi] = np.array([
                        -SinTheta[iTheta] * CosPhi[iPhi],
                        -SinTheta[iTheta] * SinPhi[iPhi], -CosTheta[iTheta], 1
                    ])
                else:
                    kVector[iEnergy][iTheta][iPhi] = np.array([
                        SinTheta[iTheta] * CosPhi[iPhi],
                        SinTheta[iTheta] * SinPhi[iPhi], CosTheta[iTheta], 1
                    ])
        kVector[iEnergy] *= k
    kVector *= -1
    print("Finish Making Wave Number Vector")

    HoloSin = np.zeros_like(Hologram)
    for iTheta in range(sizeTheta):
        HoloSin[:, iTheta, :] = SinTheta[iTheta]
    kVector *= -1
    Hologram = Hologram * HoloSin
    # AtomicImageReal = np.zeros((sizeX, sizeY, sizeZ))
    # AtomicImageImag = np.zeros((sizeX, sizeY, sizeZ))

    @jit("f8[:,:,:,:](f8[:,:,:],f8[:,:,:,:],f8[:,:,:,:],f8)", nopython=True)
    def calculateAtomicImage(Hologram, CoordinateVector, kVector, PhaseShift):
        AtomicImage = np.zeros((2, sizeX, sizeY, sizeZ))
        for iZ in range(CoordinateVector.shape[2]):
            Z = CoordinateVector[0, 0, iZ, 2]
            print(Z)
            for iX in range(CoordinateVector.shape[0]):
                # X = CoordinateVector[iX, 0, iZ, 0]
                # display = "Z : {0:.3}, X : {1:.3}".format(Z, X)
                # display = "Z : " + str(Z) + ", X : " + str(X)
                # print("\r", display, end="")
                # print(display)
                for iY in range(CoordinateVector.shape[1]):
                    if CoordinateVector[iX, iY, iZ, 3] > 1:
                        Phase = (np.tensordot(CoordinateVector[iX, iY, iZ, :],
                                              kVector,
                                              axes=((0), (3)))) + PhaseShift
                        AtomicImage[0, iX, iY,
                                    iZ] = np.tensordot(Hologram,
                                                       np.cos(Phase),
                                                       axes=((0, 1, 2), (0, 1,
                                                                         2)))
                        # AtomicImage[1, iX, iY,
                        #                 iZ] = np.tensordot(Hologram,
                        #                                    np.sin(Phase),
                        #                                    axes=((0, 1, 2),
                        #                                          (0, 1, 2)))
        return AtomicImage

    AtomicImage = calculateAtomicImage(Hologram, CoordinateVector, kVector,
                                       PhaseShift)
    AtomicImageReal = AtomicImage[0]
    AtomicImageImag = AtomicImage[1]
    print()
    fileNameReal = inputJson["InputAtomicImage"]["FileNameReal"]
    fileNameImag = inputJson["InputAtomicImage"]["FileNameImaginal"]
    writeIgorText(AtomicImageReal, outputDirectory + fileNameReal, minX, stepX,
                  minY, stepY, minZ, stepZ)
    if isCalculateImage is True:
        writeIgorText(AtomicImageImag, outputDirectory + fileNameImag, minX,
                      stepX, minY, stepY, minZ, stepZ)
    writeIgorText(-1 * AtomicImageReal,
                  outputDirectory + "Inverse" + fileNameReal, minX, stepX,
                  minY, stepY, minZ, stepZ)
    Files = {"RealPart": fileNameReal, "ImaginalPart": fileNameImag}
    jsonDict["OutputFile"] = Files
    jsonDict["Comment"] = Comment
    with open(outputDirectory + "Result" + Comment + ".json", 'w') as file:
        json.dump(jsonDict, file, indent=2)
    logdict = {
        "program": "AtomicImageReconstruction",
        "outputdirectory": outputDirectory,
        "time": CurrentTime,
        "comment": Comment
    }
    logDirectory = inputJson["LogDirectory"]
    with open(logDirectory + "XFHCalculationLog.txt", 'a') as file:
        json.dump(logdict, file, indent=2)
    return outputDirectory

def AtomicImageReconstructionReturnDirect(inputJsonFile, holograms, energys,
                              outputDirectory="",
                              isCalculateImage=True,
                              isInverseRealPart=False,
                              Comment=""):
    inputJson = json.load(open(inputJsonFile, 'r'))

    workingDir = inputJson["InputDirectory"]
    CurrentTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # if outputDirectory == "":
    #     outputDirectory = workingDir + "AtomicImageReconstruction" + CurrentTime + "/"
    #     os.mkdir(outputDirectory)
    # else:
    #     outputDirectory = inputJson["OutputDirectory"]

    jsonDict = {}
    jsonDict["Program"] = "AtomicImageReconstruction.py"
    jsonDict.update(inputJson)

    # fileList = []
    # energyList = []
    # Files = inputJson["InputHologram"]
    # for contents in Files.values():
    #     fileList.append(workingDir + contents["FileName"])
    #     energyList.append(contents["Energy"])
    # # print(fileList)
    # print(energyList)
    energyList = energys

    # Hologram = readHologramFromCSV(fileList)
    Hologram = holograms

    # print("Finish Reading File")

    minX = inputJson["MinX"]
    minY = inputJson["MinY"]
    minZ = inputJson["MinZ"]
    maxX = inputJson["MaxX"]
    maxY = inputJson["MaxY"]
    maxZ = inputJson["MaxZ"]
    stepX = inputJson["StepX"]
    stepY = inputJson["StepY"]
    stepZ = inputJson["StepZ"]

    startTheta = inputJson["StartTheta"]
    endTheta = inputJson["EndTheta"]
    stepTheta = inputJson["StepTheta"]
    startPhi = inputJson["StartPhi"]
    endPhi = inputJson["EndPhi"]
    stepPhi = inputJson["StepPhi"]

    mode = inputJson["Mode"]
    PhaseShift = inputJson["PhaseShift"] * 2 * np.pi

    sizeTheta = int((endTheta - startTheta) / stepTheta + 1)
    sizePhi = int((endPhi - startPhi) / stepPhi)
    sizeX = int((maxX - minX) / stepX + 1)
    sizeY = int((maxY - minY) / stepX + 1)
    sizeZ = int((maxZ - minZ) / stepX + 1)
    sizeEnergy = len(energyList)

    ThetaList = np.arange(startTheta, endTheta + stepTheta, stepTheta)
    PhiList = np.arange(startPhi, endPhi, stepTheta)

    CoordinateVector = np.empty((sizeX, sizeY, sizeZ, 4))
    # DistanceVector = np.empty((sizeX, sizeY, sizeZ))
    for iZ in range(sizeZ):
        z = minZ + stepZ * iZ
        for iY in range(sizeY):
            y = minY + stepY * iY
            for iX in range(sizeX):
                x = minX + stepX * iX
                Coordinate = np.array([x, y, z])
                # DistanceVector[iX][iY][iZ] = np.linalg.norm(Coordinate, ord=2)
                r = np.linalg.norm(Coordinate, ord=2)
                CoordinateVector[iX][iY][iZ] = np.append(Coordinate, r)
    # print("Finish Making Coordinate Vector")

    # set k vector k = [kx, ky, kz, -|k||]
    CosTheta = np.cos(np.deg2rad(ThetaList))
    SinTheta = np.sin(np.deg2rad(ThetaList))
    SinPhi = np.sin(np.deg2rad(PhiList))
    CosPhi = np.cos(np.deg2rad(PhiList))
    kVector = np.empty((sizeEnergy, sizeTheta, sizePhi, 4))
    for iEnergy in range(sizeEnergy):
        k = 0.505 * energyList[iEnergy] / 1000
        for iTheta in range(sizeTheta):
            for iPhi in range(sizePhi):
                if mode == "Normal":
                    kVector[iEnergy][iTheta][iPhi] = np.array([
                        -SinTheta[iTheta] * CosPhi[iPhi],
                        -SinTheta[iTheta] * SinPhi[iPhi], -CosTheta[iTheta], 1
                    ])
                else:
                    kVector[iEnergy][iTheta][iPhi] = np.array([
                        SinTheta[iTheta] * CosPhi[iPhi],
                        SinTheta[iTheta] * SinPhi[iPhi], CosTheta[iTheta], 1
                    ])
        kVector[iEnergy] *= k
    kVector *= -1
    # print("Finish Making Wave Number Vector")

    HoloSin = np.zeros_like(Hologram)
    for iTheta in range(sizeTheta):
        HoloSin[:, iTheta, :] = SinTheta[iTheta]
    kVector *= -1
    Hologram = Hologram * HoloSin
    AtomicImageReal = np.zeros((sizeX, sizeY, sizeZ))
    AtomicImageImag = np.zeros((sizeX, sizeY, sizeZ))
    for iZ in range(sizeZ):
        Z = minZ + stepZ * iZ
        for iX in range(sizeX):
            X = minX + stepX * iX
            display = "Z : {0:.3}, X : {1:.3}".format(Z, X)
            # print("\r", display, end="")
            for iY in range(sizeY):
                if CoordinateVector[iX, iY, iZ, 3] > 1:
                    Phase = (np.tensordot(CoordinateVector[iX, iY, iZ, :],
                                          kVector,
                                          axes=((0), (3)))) + PhaseShift
                    AtomicImageReal[iX, iY,
                                    iZ] = np.tensordot(Hologram,
                                                       np.cos(Phase),
                                                       axes=((0, 1, 2), (0, 1,
                                                                         2)))
                    if isCalculateImage is True:
                        AtomicImageImag[iX, iY,
                                        iZ] = np.tensordot(Hologram,
                                                           np.sin(Phase),
                                                           axes=((0, 1, 2),
                                                                 (0, 1, 2)))
    if isInverseRealPart is True:
        AtomicImageReal = AtomicImageReal * -1
    return AtomicImageReal



# %%
if __name__ == "__main__":
    Comment = input("Input Comment\n")
    AtomicImageReconstruction(
        "/Users/yuta/Desktop/test/test3/SymmetrizeHologram20210623213920/InputConfig.json",
        "", Comment)
