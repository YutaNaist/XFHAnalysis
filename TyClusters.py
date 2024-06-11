import numpy as np
import pandas as pd

try:
    from XFHAnalysis.TyHologram import TyHologram
except ModuleNotFoundError:
    from TyHologram import TyHologram


class TyClusters:

    clusters = []

    def __init__(self) -> None:
        pass

    def loadFromListDict(self, listDictClusters):
        if type(listDictClusters[0]) is dict:
            self.clusters = listDictClusters
        elif type(listDictClusters[0]) is list:
            for cluster in listDictClusters:
                self.clusters.append(cluster)

    def loadFromXyzFiles(self, listXyzFiles):
        for xyzFile in listXyzFiles:
            self.loadFromXyzFile(xyzFile)

    def loadFromXyzFile(self, xyzFile):
        count = 0
        with open(xyzFile, "r") as file:
            lines = file.readlines()
            listCluster = []
            for line in lines:
                if not (count == 0 or count == 1):
                    atom = line.split("\t")
                    dictAtom = {
                        "atom": atom[0],
                        "x": float(atom[1]),
                        "y": float(atom[2]),
                        "z": float(atom[3]),
                        "r": np.sqrt(
                            float(atom[1]) ** 2
                            + float(atom[2]) ** 2
                            + float(atom[3]) ** 2
                        ),
                    }
                    listCluster.append(dictAtom)
                count += 1
        self.clusters.append(listCluster)

    def calculateXFH(
        self,
        energyListEv=[10000],
        resolution=1,
        scatteringFactorFile=r"C:\Project\XFHAnalysis\DataBaseX-rayScattering.csv",
        measurementMode="Normal",
        phaseShift=0,
        # inputJsonFileName,
        outputDirectory="./",
        filenameBase="hologram_calc",
        AtomicFractionFileName="",
        isSaveHologram=True,
        Comment="",
        NoDisplay=False,
    ):
        def ScatteringFactor(
            ScatterName,
            ScatterParameter,
            kVector,
            Position,
            isNormal=True,
            FractionParameter={},
        ):
            scatteringFactor = np.zeros_like(kVector[:, :, :, 0])
            Lambda = 2 * np.pi / kVector[:, :, :, 3]
            Lambda2 = Lambda**2
            Cos2Theta = (
                -1
                * np.tensordot(kVector, Position, ((3), (0)))
                / (kVector[:, :, :, 3] * Position[3])
                + 1
            )
            SinScatteringAngle2 = (1 - Cos2Theta) / 2
            for i in range(4):
                scatteringFactor += (ScatterParameter[ScatterName][i * 2 + 1]) * np.exp(
                    -1
                    * SinScatteringAngle2
                    / Lambda2
                    * (ScatterParameter[ScatterName][i * 2 + 2])
                )
            scatteringFactor += ScatterParameter[ScatterName][9]
            # scatteringFactor += ScatterParameter[ScatterName][0] * 1 / (
            #     1 + SinScatteringAngle2 / Lambda2 / 0.4)
            if FractionParameter != {}:
                scatteringFactor *= np.exp(
                    -25.132736
                    * FractionParameter[ScatterName][0] ** 2
                    * SinScatteringAngle2
                    / Lambda2
                )
            # np.savetxt("scFactor.csv", scatteringFactor[0], delimiter=",")
            # np.savetxt("sin2theta.csv",
            #            SinScatteringAngle2[0] / Lambda2[0],
            #            delimiter=",")
            return scatteringFactor

        def readParameterFromCSV(filename, header=0, isDict=True, delimiter=","):
            Parameter = pd.read_csv(filename, sep=delimiter, header=header).to_dict()
            Keys = list(Parameter.keys())
            KeyLen = len(Keys)
            ValueLen = len(Parameter[Keys[0]])
            if isDict is True:
                returnDict = {}
                for i in range(ValueLen):
                    listParam = []
                    element = Parameter[Keys[0]][i]
                    for j in range(1, KeyLen):
                        listParam.append(Parameter[Keys[j]][i])
                    returnDict[element] = listParam
                return returnDict
            else:
                returnList = []
                for i in range(ValueLen):
                    listParam = []
                    element = Parameter[Keys[0]][i]
                    for j in range(KeyLen):
                        listParam.append(Parameter[Keys[j]][i])
                    returnList.append(listParam)
                return returnList

        StartTheta = 0
        EndTheta = 180
        StepTheta = resolution
        StartPhi = 0
        EndPhi = 360
        StepPhi = resolution
        EnergyList = np.array(energyListEv)
        SizeEnergy = len(EnergyList)
        if not NoDisplay:
            print(EnergyList)

        SizeTheta = int((EndTheta - StartTheta) / StepTheta + 1)
        SizePhi = int((EndPhi - StartPhi) / StepPhi)

        ThetaList = np.arange(StartTheta, EndTheta + 1, StepTheta)
        PhiList = np.arange(StartPhi, EndPhi, StepTheta)

        kVector = np.empty((SizeEnergy, SizeTheta, SizePhi, 4))
        CosTheta = np.cos(np.deg2rad(ThetaList))
        SinTheta = np.sin(np.deg2rad(ThetaList))
        SinPhi = np.sin(np.deg2rad(PhiList))
        CosPhi = np.cos(np.deg2rad(PhiList))
        isNormal = False
        for iEnergy in range(SizeEnergy):
            k = 0.505 * EnergyList[iEnergy] / 1000
            for iTheta in range(SizeTheta):
                for iPhi in range(SizePhi):
                    if measurementMode == "Normal":
                        kVector[iEnergy][iTheta][iPhi] = np.array(
                            [
                                -SinTheta[iTheta] * CosPhi[iPhi],
                                -SinTheta[iTheta] * SinPhi[iPhi],
                                -CosTheta[iTheta],
                                1,
                            ]
                        )
                        isNormal = True
                    else:
                        kVector[iEnergy][iTheta][iPhi] = np.array(
                            [
                                SinTheta[iTheta] * CosPhi[iPhi],
                                SinTheta[iTheta] * SinPhi[iPhi],
                                CosTheta[iTheta],
                                1,
                            ]
                        )
            kVector[iEnergy] *= k

        if not NoDisplay:
            print("Finish calculate k Vector")
        # DatabaseDirectory = inputJson["DatabaseDirectory"]
        ScatteringParameter = readParameterFromCSV(scatteringFactorFile, header=None)
        FractionParameter = {}
        if AtomicFractionFileName != "":
            FractionParameter = readParameterFromCSV(
                AtomicFractionFileName, header=0, isDict=True, delimiter=","
            )

        if not NoDisplay:
            print("Finish Reading ScatteringParapeter")

        calcHologram = np.zeros((SizeEnergy, SizeTheta, SizePhi))
        # for File in FileList:
        for ic, Cluster in enumerate(self.clusters):
            # if not NoDisplay:
            #     print(File)
            print("Cluster: {}".format(ic))
            # Cluster = readFromXYZFile(File)
            count = 0
            for Atom in Cluster:
                if count % 100 == 0:
                    display = "{0}/{1}".format(count, len(Cluster))
                    if not NoDisplay:
                        print("\r", display, end="")
                count += 1
                AtomicName = Atom["name"]
                AtomX = Atom["x"]
                AtomY = Atom["y"]
                AtomZ = Atom["z"]
                AtomR = Atom["r"]
                AtomicPosition = np.array([AtomX, AtomY, AtomZ, AtomR])
                scatteringFactor = (
                    ScatteringFactor(
                        AtomicName,
                        ScatteringParameter,
                        kVector,
                        AtomicPosition,
                        isNormal=isNormal,
                        FractionParameter=FractionParameter,
                    )
                    * -2.8179403227
                    * 0.00001
                )
                phase = (
                    np.tensordot(AtomicPosition, kVector, axes=((0), (3)))
                ) + phaseShift
                calcHologram += 2 * scatteringFactor / AtomR * np.cos(phase)
            if not NoDisplay:
                print()
        calcHologram = calcHologram / len(self.clusters)
        listHologram = []
        for iE, energy in enumerate(EnergyList):
            if isSaveHologram:
                np.savetxt(
                    "{}{}_{}.csv".format(outputDirectory, filenameBase, energy),
                    calcHologram[iE],
                    delimiter=",",
                )
                hologram = TyHologram(
                    "{}hologram_calc_{}.csv".format(outputDirectory, energy),
                    energy,
                    step_theta_deg=StepTheta,
                    step_phi_deg=StepPhi,
                )
                listHologram.append(hologram)
            else:
                hologram = TyHologram()
                hologram.set_hologram_from_array(
                    calcHologram[iE],
                    energy,
                    step_theta_deg=StepTheta,
                    step_phi_deg=StepPhi,
                )
                listHologram.append(hologram)
        return listHologram
