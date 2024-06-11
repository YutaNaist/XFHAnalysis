# %%
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
from scipy import optimize as opt
import re


def getIntensityFromAngle(hologram, theta, phi, stepTheta, stepPhi, sizeTheta, sizePhi):
    th = theta - theta % stepTheta
    ph = phi - phi % stepPhi
    iT = int(theta // stepTheta)
    iP = int(phi // stepPhi)

    x = convertPolerToDecalt(theta, phi)
    x0 = convertPolerToDecalt(th, ph)
    x1 = convertPolerToDecalt(th + stepTheta, ph)
    x2 = convertPolerToDecalt(th, ph + stepPhi)
    x3 = convertPolerToDecalt(th + stepTheta, ph + stepPhi)
    X0 = np.array([x0, x1, x2, x3])
    X = np.array([x, x, x, x])
    A = 1 / (np.sqrt(np.sum((X0 - X) ** 2, axis=1)) + 0.00000001)
    A = A / np.sum(A)
    partHolo = np.array(
        [
            hologram[iT, iP],
            hologram[(iT + 1) % sizeTheta, iP],
            hologram[iT, (iP + 1) % sizePhi],
            hologram[(iT + 1) % sizeTheta, (iP + 1) % sizePhi],
        ]
    )
    return np.dot(partHolo, A)


def hyperResolutionHologram(hologram, resolution, stepThetaOriginal, stepPhiOriginal):
    sizeHoloOriginal = np.shape(hologram)
    sizeThetaOriginal = sizeHoloOriginal[0]
    sizePhiOriginal = sizeHoloOriginal[1]
    sizeTheta = (sizeHoloOriginal[0] - 1) * resolution + 1
    sizePhi = sizeHoloOriginal[1] * resolution
    stepTheta = stepThetaOriginal / resolution
    stepPhi = stepPhiOriginal / resolution
    hyperResHolo = np.zeros((sizeTheta, sizePhi))
    theta = np.arange(0, np.pi + stepTheta, stepTheta)
    phi = np.arange(0, np.pi * 2, stepPhi)
    Phi, Theta = np.meshgrid(phi, theta)
    for i in range(sizeTheta):
        for j in range(sizePhi):
            hyperResHolo[i, j] = getIntensityFromAngle(
                hologram,
                Theta[i, j],
                Phi[i, j],
                stepThetaOriginal,
                stepPhiOriginal,
                sizeThetaOriginal,
                sizePhiOriginal,
            )
    np.savetxt("hyperReso.csv", hyperResHolo, delimiter=",")
    return hyperResHolo, stepTheta, stepPhi


def GramSchmitd(Direction):
    zDirection = np.array(Direction)
    zDirection = zDirection / np.linalg.norm(zDirection, 2)

    xDirection = [1, 0, 0]
    if Direction[1] == 0 and Direction[2] == 0:
        xDirection = [0, 0, 1]
    yDirection = np.cross(Direction, xDirection)

    xDirection = np.array(xDirection)
    xDirection = xDirection - np.dot(xDirection, zDirection) * zDirection
    xDirection = xDirection / np.linalg.norm(xDirection, 2)

    yDirection = np.array(yDirection)
    yDirection = yDirection - np.dot(yDirection, zDirection) * zDirection
    yDirection = yDirection - np.dot(yDirection, xDirection) * xDirection
    yDirection = yDirection / np.linalg.norm(yDirection, 2)
    return zDirection, xDirection, yDirection


def convertPolerToDecalt(theta, phi):
    a = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    return a


def convertHologramAnyDirection(
    hologram,
    stepTheta,
    stepPhi,
    direction,
    secondDirection=[0, 0, 0],
    resolution=1,
    inverse=False,
):
    print("Convert Hologram to {}".format(direction))
    sizeTheta = int(180 / stepTheta) + 1
    sizePhi = int(360 / stepTheta)
    stepThetaRad = np.radians(stepTheta)
    stepPhiRad = np.radians(stepPhi)

    convertedHologram = np.zeros((sizeTheta, sizePhi))
    sizeThetaOriginal = np.shape(hologram)[0]
    sizePhiOriginal = np.shape(hologram)[1]

    stepThetaOriginal = 180 / (sizeThetaOriginal - 1)
    stepPhiOriginal = 360 / (sizePhiOriginal)
    stepThetaRadOriginal = np.radians(stepThetaOriginal)
    stepPhiRadOriginal = np.radians(stepPhiOriginal)

    stepThetaRadHighReso = np.radians(stepThetaOriginal)
    stepPhiRadHighReso = np.radians(stepPhiOriginal)
    if resolution == 1:
        HighResohologram = hologram
    else:
        (
            HighResohologram,
            stepThetaRadHighReso,
            stepThetaRadHighReso,
        ) = hyperResolutionHologram(
            hologram, resolution, stepThetaRadOriginal, stepPhiRadOriginal
        )
    sizeThetaHighReso = HighResohologram.shape[0]
    sizePhiHighReso = HighResohologram.shape[1]

    directionNp = np.array(direction)
    direction2ndNp = np.zeros_like(directionNp)
    direction3rdNp = np.zeros_like(directionNp)
    if secondDirection == [0, 0, 0]:
        directionNp, direction2ndNp, direction3rdNp = GramSchmitd(direction)
    else:
        direction2ndNp = np.array(secondDirection)
        direction3rdNp = np.cross(directionNp, direction2ndNp)
        directionNp = directionNp / np.linalg.norm(directionNp, 2)
        direction2ndNp = direction2ndNp / np.linalg.norm(direction2ndNp, 2)
        direction3rdNp = direction3rdNp / np.linalg.norm(direction3rdNp, 2)

    theta = np.arange(0, np.pi + stepThetaRad, stepThetaRad)
    phi = np.arange(0, np.pi * 2, stepPhiRad)
    Phi, Theta = np.meshgrid(phi, theta)
    X = np.array(
        [np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)]
    )
    A = np.array([direction2ndNp, direction3rdNp, directionNp]).T
    if inverse is True:
        A = A.T
    X2 = np.tensordot(A, X, axes=(1, 0))
    ThetaConvert = np.arccos(X2[2, :, :])
    PhiConvert = np.arctan2(X2[1, :, :], X2[0, :, :]) % (2 * np.pi)

    for i in range(sizeTheta):
        for j in range(sizePhi):
            convertedHologram[i, j] = getIntensityFromAngle(
                HighResohologram,
                ThetaConvert[i, j],
                PhiConvert[i, j],
                stepThetaRadHighReso,
                stepPhiRadHighReso,
                sizeThetaHighReso,
                sizePhiHighReso,
            )
    return convertedHologram


def convertHologramAnyAngle(
    hologram,
    stepTheta,
    stepPhi,
    ZDirectionsTheta,
    ZDirectionsPhi,
    secondDirectionsTheta=-1,
    secondDirectionsPhi=-1,
    resolution=1,
    inverse=False,
):
    direction = convertPolerToDecalt(
        np.radians(ZDirectionsTheta), np.radians(ZDirectionsPhi)
    )
    secondDirection = [0, 0, 0]
    if secondDirectionsPhi != -1 and secondDirectionsTheta != -1:
        secondDirection = convertPolerToDecalt(
            np.radians(secondDirectionsTheta), np.radians(secondDirectionsPhi)
        )
    return convertHologramAnyDirection(
        hologram,
        stepTheta,
        stepPhi,
        direction,
        secondDirection=secondDirection,
        resolution=resolution,
        inverse=inverse,
    )


class Hologram:
    directory = "./"
    Hologram = np.empty((1, 1))
    Theta = np.empty(0)
    Phi = np.empty(0)
    Energy = 0
    stepTheta = 1
    stepPhi = 1

    def __init__(
        self,
        filename,
        Energy,
        isFull=True,
        startTheta=0,
        endTheta=180,
        stepTheta=1,
        startPhi=0,
        endPhi=180,
        stepPhi=1,
    ):
        self.Energy = Energy
        self.Hologram = np.loadtxt(filename, delimiter=",")
        if isFull is False:
            # resolutionTheta = (self.Hologram.shape[0] - 1) // (endTheta - startTheta)
            # resolutionPhi = (self.Hologram.shape[1]) // (endPhi - startPhi)
            HoloTemp = np.zeros((int(180 / stepTheta) + 1, int(360 / stepPhi)))
            indexTheta = 0
            indexPhi = 0
            for i in range(HoloTemp.shape[0]):
                if (
                    startTheta / stepTheta - 0.00000001
                    < i
                    < endTheta / stepTheta + 1.00000001
                ):
                    indexPhi = 0
                    for j in range(HoloTemp.shape[0]):
                        if (
                            startPhi / stepPhi - 0.00000001
                            < j
                            < endPhi / stepPhi + 0.00000001
                        ):
                            HoloTemp[i, j] = self.Hologram[indexTheta, indexPhi]
                            indexPhi += 1
                    indexTheta += 1
            self.Hologram = HoloTemp
        sizeTheta = self.Hologram.shape[0]
        sizePhi = self.Hologram.shape[1]
        stepTheta = 180 / (sizeTheta - 1)
        stepPhi = 360 / sizePhi
        self.Theta = np.deg2rad(np.arange(0, 180 + stepTheta, stepTheta))
        self.Phi = np.deg2rad(np.arange(0, 360, stepPhi))
        self.stepTheta = stepTheta
        self.stepPhi = stepPhi
        dirSplit = filename.split("/")
        if len(dirSplit) != 1:
            self.directory = ""
            for i in range(len(dirSplit) - 1):
                self.directory += dirSplit[i] + "/"

    def get_intensity_from_angle(self, theta, phi):
        theta = theta % 180
        phi = phi % 360
        sizeTheta = self.Hologram.shape[0]
        sizePhi = self.Hologram.shape[1]
        HoloTemp = np.zeros((sizeTheta, sizePhi + 1))
        HoloTemp[:, :-1] = self.Hologram
        HoloTemp[:, -1] = self.Hologram[:, 0]
        thetaIndex = int(theta / self.stepTheta)
        phiIndex = int(phi / self.stepPhi)
        theta = (theta - np.rad2deg(self.Theta[thetaIndex])) / self.stepTheta
        phi = (phi - np.rad2deg(self.Phi[phiIndex])) / self.stepPhi
        intensity = 0
        intensity += HoloTemp[thetaIndex, phiIndex] * (1 - theta) * (1 - phi)
        intensity += HoloTemp[thetaIndex + 1, phiIndex] * (theta) * (1 - phi)
        intensity += HoloTemp[thetaIndex, phiIndex + 1] * (1 - theta) * (phi)
        intensity += HoloTemp[thetaIndex + 1, phiIndex + 1] * (theta) * (phi)
        return intensity

    def changeResolution(self, stepTheta, stepPhi):
        resolutionTheta = self.stepTheta / stepTheta
        resolutionPhi = self.stepPhi / stepPhi
        thetaTemp = np.arange(0, 180 + stepTheta, stepTheta)
        phiTemp = np.arange(0, 360, stepPhi)
        HoloTemp = np.zeros((len(thetaTemp), len(phiTemp)))
        if resolutionTheta >= 1 and resolutionPhi >= 1:
            for i in range(len(thetaTemp)):
                for j in range(len(phiTemp)):
                    HoloTemp[i, j] = self.intensityFromAngle(thetaTemp[i], phiTemp[j])
        else:
            intResolutionTheta = resolutionTheta // 1
            intResolutionPhi = resolutionPhi // 1
            decResolutionTheta = resolutionTheta % 1
            decResolutionPhi = resolutionPhi % 1
            for i in range(len(thetaTemp)):
                theta = stepTheta * i
                for j in range(len(phiTemp)):
                    phi = stepPhi * j
                    intensity = 0
                    for m in range(int(intResolutionTheta)):
                        for n in range(int(intResolutionPhi)):
                            intensity += self.intensityFromAngle(
                                theta + stepTheta * m, phi + stepPhi * n
                            )
                    for m in range(int(intResolutionTheta)):
                        intensity += (
                            self.intensityFromAngle(
                                theta + stepTheta * m, phi + stepPhi * intResolutionPhi
                            )
                            * decResolutionPhi
                        )
                    for n in range(int(intResolutionPhi)):
                        intensity += (
                            self.intensityFromAngle(
                                theta + stepTheta * decResolutionTheta,
                                phi + stepPhi * n,
                            )
                            * decResolutionTheta
                        )
                    intensity += (
                        self.intensityFromAngle(
                            theta + stepTheta * intResolutionTheta,
                            phi + stepPhi * intResolutionPhi,
                        )
                        * decResolutionTheta
                        * decResolutionPhi
                    )
                    HoloTemp[i, j] = intensity
        self.Theta = np.deg2rad(thetaTemp)
        self.Phi = np.deg2rad(phiTemp)
        self.Hologram = HoloTemp
        self.stepTheta = stepTheta
        self.stepPhi = stepPhi
        return 0

    def saveHologram(self, filename):
        np.savetxt(self.directory + filename, self.Hologram, delimiter=",")

    def indexTheta(self, theta):
        for i, angle in enumerate(self.Theta):
            if np.abs(theta - angle) < 0.0001:
                return i
        return 0

    def indexPhi(self, phi):
        for i, angle in enumerate(self.Phi):
            if np.abs(phi - angle) < 0.0001:
                return i
        return 0

    def show1D(self, iTheta, isInverse=False, isSave=False):
        if isInverse is False:
            plt.plot(self.Phi, self.Hologram[iTheta])
        else:
            plt.plot(self.Theta, self.Hologram[:, iTheta])
        if isSave is True:
            plt.savefig(self.directory + "Hologram1D.png")
        plt.show()

    def show1DWithFit(self, iTheta, fitFunction, isInverse=False, isSave=False):
        if isInverse is False:
            plt.plot(self.Phi, self.Hologram[iTheta])
            plt.plot(self.Phi, fitFunction)
        else:
            plt.plot(self.Theta, self.Hologram[:, iTheta])
            plt.plot(self.Theta, fitFunction)
        if isSave is True:
            plt.savefig(self.directory + "Hologram1D.png")
        plt.show()

    def show2D(
        self,
        outputDirectory="",
        color="Greys_r",
        minimum=None,
        maximum=None,
        isImshow=True,
    ):
        if minimum is None:
            minimum = np.min(self.Hologram)
        if maximum is None:
            maximum = np.max(self.Hologram)
        plt.imshow(
            self.Hologram,
            cmap=color,
            vmin=minimum,
            vmax=maximum,
            extent=[
                np.rad2deg(self.Phi[0]),
                np.rad2deg(self.Phi[-1]),
                np.rad2deg(self.Theta[0]),
                np.rad2deg(self.Theta[-1]),
            ],
        )
        if outputDirectory == "":
            outputDirectory = self.directory
        plt.savefig(outputDirectory + "Hologram2D{}.png".format(self.Energy))
        if isImshow:
            plt.show()

    def save2D(self, color="Greys_r", minimum=None, maximum=None):
        if minimum is None:
            minimum = np.min(self.Hologram)
        if maximum is None:
            maximum = np.max(self.Hologram)
        plt.imsave(
            self.directory + "Hologram2D.png",
            self.Hologram,
            cmap=color,
            vmin=minimum,
            vmax=maximum,
        )

    def show3D(self):
        sizeTheta = self.Hologram.shape[0]
        sizePhi = self.Hologram.shape[1]
        HoloTemp = np.zeros((sizeTheta, sizePhi + 1))
        HoloTemp[:, :-1] = self.Hologram
        HoloTemp[:, -1] = self.Hologram[:, 0]
        Phi = np.append(self.Phi, np.deg2rad(360))
        r = 1
        x = r * np.outer(np.sin(self.Theta), np.cos(Phi).T)
        y = r * np.outer(np.sin(self.Theta), np.sin(Phi).T)
        z = r * np.outer(np.cos(self.Theta), np.ones_like(Phi).T)

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        mlab.mesh(x, y, z, scalars=HoloTemp, colormap="gray")
        mlab.view(0, 0, 5, (0, 0, 0))
        mlab.show()
        print("finish")

    def save3D(self, outputDirectory=""):
        sizeTheta = self.Hologram.shape[0]
        sizePhi = self.Hologram.shape[1]
        HoloTemp = np.zeros((sizeTheta, sizePhi + 1))
        HoloTemp[:, :-1] = self.Hologram
        HoloTemp[:, -1] = self.Hologram[:, 0]
        Phi = np.append(self.Phi, np.deg2rad(360))
        r = 1
        x = r * np.outer(np.sin(self.Theta), np.cos(Phi).T)
        y = r * np.outer(np.sin(self.Theta), np.sin(Phi).T)
        z = r * np.outer(np.cos(self.Theta), np.ones_like(Phi).T)

        # f = mlab.gcf()
        # camera = f.scene.camera
        # camera.yaw(-45)
        # camera.pitch(35.6 - 90)
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        # mlab.clf()
        mlab.mesh(x, y, z, scalars=HoloTemp, colormap="gray")
        # mlab.view(0, 0, 0, (0, 0, 0))
        mlab.view(0, 0, 5, (0, 0, 0))
        mlab.savefig(self.directory + "Hologram3D{}XY.png".format(self.Energy))
        mlab.view(0, 90, 5, (0, 0, 0))
        mlab.savefig(self.directory + "Hologram3D{}YZ.png".format(self.Energy))
        mlab.view(90, 90, 5, (0, 0, 0))
        if outputDirectory == "":
            outputDirectory = self.directory
        mlab.savefig(outputDirectory + "Hologram3D{}XZ.png".format(self.Energy))
        mlab.show()
        print("finish")

    def FourierLine(self, iTheta, r):
        FourierReal = np.zeros_like(r)
        FourierImag = np.zeros_like(r)
        for iPhi in range(self.Phi.shape[0]):
            FourierReal += self.Hologram[iTheta, iPhi] * np.cos(self.Phi[iPhi] * r)
            FourierImag += self.Hologram[iTheta, iPhi] * np.sin(self.Phi[iPhi] * r)
        return FourierReal, FourierImag

    def fittingPolynomial(self, iTheta, PolynomialDimension, isPhi=False):
        def fittingFunc(x, *parameter):
            y = 0
            for i in range(PolynomialDimension + 1):
                y += parameter[i] * x**i
            return y

        parameter = []
        initial = [0 for i in range(PolynomialDimension + 1)]
        if isPhi is False:
            parameter, error = opt.curve_fit(
                fittingFunc, self.Phi, self.Hologram[iTheta], p0=initial
            )
            return fittingFunc(self.Phi, *parameter)
        else:
            parameter, error = opt.curve_fit(
                fittingFunc, self.Theta, self.Hologram[:iTheta], p0=initial
            )
            return fittingFunc(self.Theta, *parameter)

    def fittingGaussian(
        self, iTheta, PolynomialDimension, PeakNum, initialGaussian, isPhi=True
    ):
        def fittingFunc(x, *parameter):
            y = 0
            for i in range(PolynomialDimension + 1):
                y += parameter[i] * x**i
            for i in range(PeakNum):
                y += parameter[PolynomialDimension + i * 3 + 1] * np.exp(
                    -1
                    * (x - parameter[PolynomialDimension + i * 3 + 2]) ** 2
                    / parameter[PolynomialDimension + i * 3 + 3]
                )
            return y

        initial = [0 for i in range(PolynomialDimension + 1)]
        initial *= initialGaussian
        if isPhi is False:
            parameter, error = opt.curve_fit(
                fittingFunc, self.Phi, self.Hologram[iTheta], p0=initial
            )
            return fittingFunc(self.Phi, *parameter)
        else:
            parameter, error = opt.curve_fit(
                fittingFunc, self.Theta, self.Hologram[:iTheta], p0=initial
            )
            return fittingFunc(self.Theta, *parameter)

    def divideByFunction(self, iTheta, Function, isPhi=False):
        if isPhi is False:
            self.Hologram[iTheta] = self.Hologram[iTheta] / Function
        else:
            self.Hologram[:, iTheta] = self.Hologram[:, iTheta] / Function

    def convertOtherDirection(self, DirectionZ=[0, 0, 1], DirectionX=""):
        holo = self.Hologram
        convertedHolo = convertHologramAnyDirection(
            hologram=holo,
            stepTheta=self.stepTheta,
            stepPhi=self.stepPhi,
            direction=DirectionZ,
            secondDirection=DirectionX,
        )
        return convertedHolo

    def symmetrizeHologram(self, symmetry="x1y1z1"):
        phiSize = self.Hologram.shape(1)
        if re.search("[0-9]z", symmetry):
            SymmetrizedHologram = self.Hologram.copy()
            rotationSymmetry = int(re.search("[0-9]z", symmetry).group()[0])
            for i in range(rotationSymmetry):
                SymmetrizedHologram += np.roll(
                    self.Hologram, phiSize // rotationSymmetry * i, axis=1
                ).copy()
            self.Hologram = SymmetrizedHologram.copy() / rotationSymmetry
        if re.search("[0-9]x", symmetry):
            SymmetrizedHologram = Hologram.copy()
            rotationSymmetry = int(re.search("[0-9]x", symmetry).group()[0])
            for i in range(rotationSymmetry):
                ConvertedHologram = self.convertOtherDirection([1, 0, 0], [0, 0, 1])
                SymmetrizedHologram += np.roll(
                    ConvertedHologram, phiSize // rotationSymmetry * i, axis=1
                ).copy()
            self.Hologram = SymmetrizedHologram.copy() / rotationSymmetry
        if re.search("[0-9]y", symmetry):
            SymmetrizedHologram = Hologram.copy()
            rotationSymmetry = int(re.search("[0-9]y", symmetry).group()[0])
            for i in range(rotationSymmetry):
                ConvertedHologram = self.convertOtherDirection([1, 0, 0], [0, 0, 1])
                SymmetrizedHologram += np.roll(
                    ConvertedHologram, phiSize // rotationSymmetry * i, axis=1
                ).copy()
            self.Hologram = SymmetrizedHologram.copy() / rotationSymmetry
        if "mz" in symmetry:
            SymmetrizedHologram += np.flip(self.Hologram, axis=0).copy()
            self.Hologram = SymmetrizedHologram.copy() / 2
        if "mx" in symmetry:
            SymmetrizedHologram += np.flip(self.Hologram, axis=1).copy()
            self.Hologram = SymmetrizedHologram.copy() / 2
        if "my" in symmetry:
            SymmetrizedHologram += np.flip(
                np.roll(self.Hologram, phiSize // 2, axis=1).copy(), axis=1
            ).copy()
            self.Hologram = SymmetrizedHologram.copy() / 2


# %%
if __name__ == "__main__":
    directory = "/Users/yuta/Desktop/ImageShift/CuVer2/CalculationPair/XFHCalculation20201110040914/"
    holo = Hologram(directory + "Hologram17500.csv", 17500)
    r = np.arange(-5, 5 + 0.1, 0.1)
    holo.changeResolution(0.25, 0.25)
    holo.show3D()
    # for i in range(91):
    #     if i % 10 == 0:
    #         print(i)
    #         FourierReal, FourierImag = holo.FourierLine(i, r)
    #         plt.plot(r, FourierReal)
    #         plt.plot(r, FourierImag)
    #         plt.show()
    # plt.plot(np.rad2deg(holo.Phi), holo.Hologram[91])
    # plt.show()

    # holo.save3D()
    # holo = Hologram(directory + "Hologram18000.csv", 18000)
    # holo.save3D()
    # holo = Hologram(directory + "Hologram18500.csv", 18500)
    # holo.save3D()
    # holo = Hologram(directory + "Hologram19000.csv", 19000)
    # holo.save3D()
    # holo = Hologram(directory + "Hologram19500.csv", 19500)
    # holo.save3D()
    # holo = Hologram(directory + "Hologram20000.csv", 20000)
    # holo.save3D()

# %%

# %%
