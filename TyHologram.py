# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy import interpolate as interp
import re


class TyHologram:
    directory = "./"
    Hologram = np.empty((1, 1))
    theta = np.empty(0)
    phi = np.empty(0)
    energy = 0
    step_theta = 1
    step_phi = 1

    def __init__(
        self,
        filename: str = "",
        energy: float = 10000,
        start_theta_deg: float = 0,
        end_theta_deg: float = 180,
        step_theta_deg: float = 1,
        start_phi_deg: float = 0,
        end_phi_deg: float = 360,
        step_phi_deg: float = 1,
    ) -> None:
        self.energy = energy
        self.step_deg = step_theta_deg
        if step_theta_deg > step_phi_deg:
            self.step_deg = step_phi_deg
        self.step_rad = np.deg2rad(self.step_deg)

        self.theta_deg = np.arange(0, 180 + step_theta_deg, step_theta_deg)
        self.phi_deg = np.arange(0, 360, step_phi_deg)
        self.theta_rad = np.deg2rad(self.theta_deg)
        self.phi_rad = np.deg2rad(self.phi_deg)
        self.Hologram = np.zeros(
            (int(180 / self.step_deg) + 1, int(360 / self.step_deg))
        )
        self.shape = self.Hologram.shape
        self.directory = ""

        if filename != "":
            self.load_from_file(
                filename,
                start_theta_deg,
                end_theta_deg,
                step_theta_deg,
                start_phi_deg,
                end_phi_deg,
                step_phi_deg,
            )

    def load_from_file(
        self,
        filename: str,
        energy: float = 10000,
        start_theta_deg: float = 0,
        end_theta_deg: float = 180,
        step_theta_deg: float = 1,
        start_phi_deg: float = 0,
        end_phi_deg: float = 360,
        step_phi_deg: float = 1,
    ) -> None:
        hologram_load = np.loadtxt(filename, delimiter=",")
        theta_load = np.arange(
            start_theta_deg, end_theta_deg + step_theta_deg, step_theta_deg
        )
        phi_load = np.arange(start_phi_deg, end_phi_deg, step_phi_deg)
        if theta_load.shape[0] != hologram_load.shape[0]:
            print("theta size is not correct")
            raise ValueError
        elif phi_load.shape[0] != hologram_load.shape[1]:
            print("phi size is not correct")
            raise ValueError
        if step_theta_deg > step_phi_deg:
            for j in range(self.Hologram.shape[1]):
                function = interp.interp1d(
                    theta_load, hologram_load[:, j], kind="cubic"
                )
                for i in range(self.Hologram.shape[0]):
                    theta = self.theta_deg[i]
                    if theta >= start_theta_deg and theta <= end_theta_deg:
                        self.Hologram[i, j] = function(theta)
                        if self.Hologram[i, j] == 0:
                            self.Hologram[i, j] = 10**-20
        elif step_theta_deg < step_phi_deg:
            for i in range(self.Hologram.shape[0]):
                function = interp.interp1d(phi_load, hologram_load[i, :], kind="cubic")
                for j in range(self.Hologram.shape[1]):
                    phi = self.phi_deg[j]
                    if phi >= start_phi_deg and theta <= end_phi_deg:
                        self.Hologram[i, j] = function(phi)
                        if self.Hologram[i, j] == 0:
                            self.Hologram[i, j] = 10**-20
        else:
            start_theta_index = int(start_theta_deg / step_theta_deg)
            start_phi_index = int(start_phi_deg / step_phi_deg)
            for i in range(hologram_load.shape[0]):
                for j in range(hologram_load.shape[1]):
                    self.Hologram[i + start_theta_index, j + start_phi_index] = (
                        hologram_load[i, j]
                    )
                    if self.Hologram[i + start_theta_index, j + start_phi_index] == 0:
                        self.Hologram[i + start_theta_index, j + start_phi_index] = (
                            10**-20
                        )
        dirSplit = filename.split("/")
        if len(dirSplit) != 1:
            self.directory = ""
            for i in range(len(dirSplit) - 1):
                self.directory += dirSplit[i] + "/"

    def set_hologram_from_array(
        self,
        hologram: np.ndarray,
        energy: float = 10000,
        save_path: str = "./",
        start_theta_deg: float = 0,
        end_theta_deg: float = 180,
        step_theta_deg: float = 1,
        start_phi_deg: float = 0,
        end_phi_deg: float = 360,
        step_phi_deg: float = 1,
    ) -> None:
        hologram_load = hologram
        theta_load = np.arange(
            start_theta_deg, end_theta_deg + step_theta_deg, step_theta_deg
        )
        phi_load = np.arange(start_phi_deg, end_phi_deg, step_phi_deg)
        if theta_load.shape[0] != hologram_load.shape[0]:
            print("theta size is not correct")
            raise ValueError
        elif phi_load.shape[0] != hologram_load.shape[1]:
            print("phi size is not correct")
            raise ValueError
        if step_theta_deg > step_phi_deg:
            for j in range(self.Hologram.shape[1]):
                function = interp.interp1d(
                    theta_load, hologram_load[:, j], kind="cubic"
                )
                for i in range(self.Hologram.shape[0]):
                    theta = self.theta_deg[i]
                    if theta >= start_theta_deg and theta <= end_theta_deg:
                        self.Hologram[i, j] = function(theta)
                        if self.Hologram[i, j] == 0:
                            self.Hologram[i, j] = 10**-20
        elif step_theta_deg < step_phi_deg:
            for i in range(self.Hologram.shape[0]):
                function = interp.interp1d(phi_load, hologram_load[i, :], kind="cubic")
                for j in range(self.Hologram.shape[1]):
                    phi = self.phi_deg[j]
                    if phi >= start_phi_deg and theta <= end_phi_deg:
                        self.Hologram[i, j] = function(phi)
                        if self.Hologram[i, j] == 0:
                            self.Hologram[i, j] = 10**-20
        else:
            start_theta_index = int(start_theta_deg / step_theta_deg)
            start_phi_index = int(start_phi_deg / step_phi_deg)
            for i in range(hologram_load.shape[0]):
                for j in range(hologram_load.shape[1]):
                    self.Hologram[i + start_theta_index, j + start_phi_index] = (
                        hologram_load[i, j]
                    )
                    if self.Hologram[i + start_theta_index, j + start_phi_index] == 0:
                        self.Hologram[i + start_theta_index, j + start_phi_index] = (
                            10**-20
                        )
        if save_path != "./":
            self.directory = save_path

    def get_intensity_from_angle_radian(
        self, theta_rad: float, phi_rad: float
    ) -> float:
        return self.get_intensity_from_angle(
            theta_deg=np.rad2deg(theta_rad), phi_deg=np.rad2deg(phi_rad)
        )

    def get_intensity_from_angle(self, theta_deg: float, phi_deg: float) -> float:
        # print(theta_deg, phi_deg)
        theta_deg = theta_deg % 180
        phi_deg = phi_deg % 360
        size_theta = self.Hologram.shape[0]
        size_phi = self.Hologram.shape[1]
        HoloTemp = np.zeros((size_theta, size_phi + 2))
        HoloTemp[:, :-2] = self.Hologram
        HoloTemp[:, -2] = self.Hologram[:, 0]
        HoloTemp[:, -1] = self.Hologram[:, 1]
        phi_deg_extend = np.zeros(size_phi + 2)
        phi_deg_extend[:-2] = self.phi_deg
        phi_deg_extend[-2] = self.phi_deg[0] + 360
        phi_deg_extend[-1] = self.phi_deg[1] + 360
        # print(theta_deg, phi_deg)
        theta_index = int(theta_deg / self.step_deg)
        phi_index = int(phi_deg / self.step_deg)
        # print(theta_index, phi_index)
        # print()
        theta_distance = (theta_deg - self.theta_deg[theta_index]) / self.step_deg
        phi_distance = (phi_deg - phi_deg_extend[phi_index]) / self.step_deg
        intensity = 0
        intensity += (
            HoloTemp[theta_index, phi_index] * (1 - theta_distance) * (1 - phi_distance)
        )
        intensity += (
            HoloTemp[theta_index + 1, phi_index] * (theta_distance) * (1 - phi_distance)
        )
        intensity += (
            HoloTemp[theta_index, phi_index + 1] * (1 - theta_distance) * (phi_distance)
        )
        intensity += (
            HoloTemp[theta_index + 1, phi_index + 1] * (theta_distance) * (phi_distance)
        )
        return intensity

    def change_resolution_hologram(self, step_deg: float, is_update_self: bool = False):
        thetaTemp = np.arange(0, 180 + step_deg, step_deg)
        phiTemp = np.arange(0, 360, step_deg)
        HoloTemp = np.zeros((len(thetaTemp), len(phiTemp)))
        if step_deg <= self.step_deg:
            for i in range(len(thetaTemp)):
                for j in range(len(phiTemp)):
                    HoloTemp[i, j] = self.intensityFromAngle(thetaTemp[i], phiTemp[j])
        else:
            int_search_range = int(step_deg / self.step_deg)
            for i in range(len(thetaTemp)):
                theta = step_deg * i
                for j in range(len(phiTemp)):
                    phi = step_deg * j
                    intensity += 0
                    weight = 0
                    for m in range(int_search_range):
                        for n in range(int_search_range):
                            if (m == 0) and (n == 0):
                                intensity += self.intensityFromAngle(theta, phi)
                                weight += 1
                            elif m == 0:
                                intensity += (
                                    self.intensityFromAngle(theta, phi + n * step_deg)
                                    * (int_search_range + 1 - n)
                                    / (int_search_range + 1)
                                )
                                intensity += (
                                    self.intensityFromAngle(theta, phi - n * step_deg)
                                    * (int_search_range + 1 - n)
                                    / (int_search_range + 1)
                                )
                                weight += (
                                    2
                                    * (int_search_range + 1 - n)
                                    / (int_search_range + 1)
                                )
                            elif n == 0:
                                intensity += (
                                    self.intensityFromAngle(theta + m * step_deg, phi)
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                                intensity += (
                                    self.intensityFromAngle(theta - m * step_deg, phi)
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                                weight += (
                                    2
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                            else:
                                intensity += (
                                    self.intensityFromAngle(
                                        theta + m * step_deg, phi + n * step_deg
                                    )
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                                intensity += (
                                    self.intensityFromAngle(
                                        theta - m * step_deg, phi + n * step_deg
                                    )
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                                intensity += (
                                    self.intensityFromAngle(
                                        theta + m * step_deg, phi - n * step_deg
                                    )
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                                intensity += (
                                    self.intensityFromAngle(
                                        theta - m * step_deg, phi - n * step_deg
                                    )
                                    * (int_search_range + 1 - m)
                                    / (int_search_range + 1)
                                )
                                weight += (
                                    4
                                    * (int_search_range + 1 - m)
                                    * (int_search_range + 1 - n)
                                    / (int_search_range + 1) ** 2
                                )
                    HoloTemp[i, j] = intensity
        return_Hologram = TyHologram()
        return_Hologram.set_hologram_from_array(
            HoloTemp,
            save_path=self.directory,
            step_theta_deg=step_deg,
            step_phi_deg=step_deg,
        )
        if is_update_self:
            self.Hologram = HoloTemp
            self.shape = self.Hologram.shape
            self.theta_deg = thetaTemp
            self.phi_rdeg = phiTemp
            self.theta_rad = np.deg2rad(thetaTemp)
            self.phi_rad = np.deg2rad(phiTemp)
            self.step_deg = step_deg
            self.step_rad = np.deg2rad(step_deg)
        return return_Hologram

    def saveHologram(self, filename):
        np.savetxt(self.directory + filename, self.Hologram, delimiter=",")

    def index_theta(self, theta_deg):
        min_distance = 10000
        min_index = 0
        for i, angle in enumerate(self.theta_deg):
            distance = np.abs(theta_deg - angle)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return min_index

    def index_phi(self, phi_deg):
        min_distance = 10000
        min_index = 0
        for i, angle in enumerate(self.phi_deg):
            distance = np.abs(phi_deg - angle)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return min_index

    def show1D(
        self,
        index_theta,
        output_directory="",
        is_inverse=False,
        is_display=True,
        is_save=False,
        filename="hologram_1d.png",
    ):
        if output_directory == "":
            output_directory = self.directory
        if is_inverse is False:
            plt.plot(self.phi_deg, self.Hologram[index_theta])
        else:
            plt.plot(self.theta_deg, self.Hologram[:, index_theta])
        if is_save is True:
            plt.savefig(self.directory + filename)
        if is_display is True:
            plt.show()
        plt.close()

    def show1DWithFit(
        self,
        index_theta,
        fitFunction,
        output_directory="",
        is_inverse=False,
        is_display=True,
        is_save=False,
        filename="hologram_1d_fit.png",
    ):
        if output_directory == "":
            output_directory = self.directory
        if is_inverse is False:
            plt.plot(self.phi_deg, self.Hologram[index_theta])
            plt.plot(self.phi_deg, fitFunction)
        else:
            plt.plot(self.theta_deg, self.Hologram[:, index_theta])
            plt.plot(self.theta_deg, fitFunction)
        if is_save is True:
            plt.savefig(self.directory + filename)
        if is_display is True:
            plt.show()
        plt.close()

    def show2D(
        self,
        output_directory="",
        filename="hologram_2d.png",
        color="Greys_r",
        minimum=None,
        maximum=None,
        is_display=True,
        is_save=False,
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
                self.phi_deg[0],
                self.phi_deg[-1],
                self.theta_deg[0],
                self.theta_deg[-1],
            ],
            origin="lower",
        )
        if output_directory == "":
            output_directory = self.directory
        if is_save:
            plt.savefig(output_directory + filename)
        if is_display:
            plt.show()
        plt.close()

    def show3D(
        self,
        colormap="gray",
        output_directory="",
        filename="hologram_3d.png",
        is_save=False,
    ):
        try:
            from mayavi import mlab
        except ImportError:
            print("Mayavi is not installed")
            raise ImportError
        sizetheta = self.Hologram.shape[0]
        sizephi = self.Hologram.shape[1]
        HoloTemp = np.zeros((sizetheta, sizephi + 1))
        HoloTemp[:, :-1] = self.Hologram
        HoloTemp[:, -1] = self.Hologram[:, 0]
        phi = np.append(self.phi, np.deg2rad(360))
        r = 1
        x = r * np.outer(np.sin(self.theta), np.cos(phi).T)
        y = r * np.outer(np.sin(self.theta), np.sin(phi).T)
        z = r * np.outer(np.cos(self.theta), np.ones_like(phi).T)

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        mlab.mesh(x, y, z, scalars=HoloTemp, colormap=colormap)
        mlab.view(0, 0, 5, (0, 0, 0))
        if outputDirectory == "":
            outputDirectory = self.directory
        if is_save:
            mlab.savefig(output_directory + filename)
        mlab.show()
        print("finish")

    def FourierLine(self, itheta, r):
        FourierReal = np.zeros_like(r)
        FourierImag = np.zeros_like(r)
        for iphi in range(self.phi.shape[0]):
            FourierReal += self.Hologram[itheta, iphi] * np.cos(self.phi[iphi] * r)
            FourierImag += self.Hologram[itheta, iphi] * np.sin(self.phi[iphi] * r)
        return FourierReal, FourierImag

    def fittingPolynomial(self, itheta, PolynomialDimension, isphi=False):
        def fittingFunc(x, *parameter):
            y = 0
            for i in range(PolynomialDimension + 1):
                y += parameter[i] * x**i
            return y

        parameter = []
        initial = [0 for i in range(PolynomialDimension + 1)]
        if isphi is False:
            parameter, error = opt.curve_fit(
                fittingFunc, self.phi, self.Hologram[itheta], p0=initial
            )
            return fittingFunc(self.phi, *parameter)
        else:
            parameter, error = opt.curve_fit(
                fittingFunc, self.theta, self.Hologram[:itheta], p0=initial
            )
            return fittingFunc(self.theta, *parameter)

    def fittingGaussian(
        self, itheta, PolynomialDimension, PeakNum, initialGaussian, isphi=True
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
        if isphi is False:
            parameter, error = opt.curve_fit(
                fittingFunc, self.phi, self.Hologram[itheta], p0=initial
            )
            return fittingFunc(self.phi, *parameter)
        else:
            parameter, error = opt.curve_fit(
                fittingFunc, self.theta, self.Hologram[:itheta], p0=initial
            )
            return fittingFunc(self.theta, *parameter)

    def divideByFunction(self, itheta, Function, isphi=False):
        if isphi is False:
            self.Hologram[itheta] = self.Hologram[itheta] / Function
        else:
            self.Hologram[:, itheta] = self.Hologram[:, itheta] / Function

    def convertOtherDirection(self, DirectionZ=[0, 0, 1], DirectionX=""):
        def getIntensityFromAngle(
            hologram, theta, phi, stepTheta, stepPhi, sizeTheta, sizePhi
        ):
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

        def hyperResolutionHologram(
            hologram, resolution, stepThetaOriginal, stepPhiOriginal
        ):
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
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            return a

        def convertHologramAnyDirection(
            hologram,
            step_theta,
            step_phi,
            direction,
            secondDirection=[0, 0, 0],
            resolution=1,
            inverse=False,
        ):
            print("Convert Hologram to {}".format(direction))
            sizetheta = int(180 / step_theta) + 1
            sizephi = int(360 / step_theta)
            step_thetaRad = np.radians(step_theta)
            step_phiRad = np.radians(step_phi)

            convertedHologram = np.zeros((sizetheta, sizephi))
            sizethetaOriginal = np.shape(hologram)[0]
            sizephiOriginal = np.shape(hologram)[1]

            step_thetaOriginal = 180 / (sizethetaOriginal - 1)
            step_phiOriginal = 360 / (sizephiOriginal)
            step_thetaRadOriginal = np.radians(step_thetaOriginal)
            step_phiRadOriginal = np.radians(step_phiOriginal)

            step_thetaRadHighReso = np.radians(step_thetaOriginal)
            step_phiRadHighReso = np.radians(step_phiOriginal)
            if resolution == 1:
                HighResohologram = hologram
            else:
                (
                    HighResohologram,
                    step_thetaRadHighReso,
                    step_thetaRadHighReso,
                ) = hyperResolutionHologram(
                    hologram, resolution, step_thetaRadOriginal, step_phiRadOriginal
                )
            sizethetaHighReso = HighResohologram.shape[0]
            sizephiHighReso = HighResohologram.shape[1]

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

            theta = np.arange(0, np.pi + step_thetaRad, step_thetaRad)
            phi = np.arange(0, np.pi * 2, step_phiRad)
            phi, theta = np.meshgrid(phi, theta)
            X = np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            A = np.array([direction2ndNp, direction3rdNp, directionNp]).T
            if inverse is True:
                A = A.T
            X2 = np.tensordot(A, X, axes=(1, 0))
            thetaConvert = np.arccos(X2[2, :, :])
            phiConvert = np.arctan2(X2[1, :, :], X2[0, :, :]) % (2 * np.pi)

            for i in range(sizetheta):
                for j in range(sizephi):
                    convertedHologram[i, j] = getIntensityFromAngle(
                        HighResohologram,
                        thetaConvert[i, j],
                        phiConvert[i, j],
                        step_thetaRadHighReso,
                        step_phiRadHighReso,
                        sizethetaHighReso,
                        sizephiHighReso,
                    )
            return convertedHologram

        holo = self.Hologram
        convertedHolo = convertHologramAnyDirection(
            hologram=holo,
            step_theta=self.step_theta,
            step_phi=self.step_phi,
            direction=DirectionZ,
            secondDirection=DirectionX,
        )
        return convertedHolo

    def symmetrize_hologram(self, symmetry="x1y1z1", is_update_self: bool = False):
        phiSize = self.Hologram.shape[1]
        Hologram_Temp = np.copy(self.Hologram)
        duplicate_symmetry = 1
        if re.search("[0-9]z", symmetry):
            SymmetrizedHologram = np.zeros_like(Hologram_Temp)
            rotationSymmetry = int(re.search("[0-9]z", symmetry).group()[0])
            for i in range(rotationSymmetry):
                SymmetrizedHologram += np.roll(
                    self.Hologram, phiSize // rotationSymmetry * i, axis=1
                ).copy()
            Hologram_Temp = SymmetrizedHologram.copy()
            duplicate_symmetry *= rotationSymmetry
        if re.search("[0-9]x", symmetry):
            # SymmetrizedHologram = np.copy(Hologram_Temp)
            rotationSymmetry = int(re.search("[0-9]x", symmetry).group()[0])
            rotated_hologram = self.rotate_hologram_euler_angle(0, 90, 0, False)
            rotated_hologram.symmetrize_hologram("z" + str(rotationSymmetry), True)
            rotated_hologram.rotate_hologram_euler_angle(0, -90, 0)
            Hologram_Temp = rotated_hologram.Hologram.copy()
            # duplicate_symmetry *= rotationSymmetry
        if re.search("[0-9]y", symmetry):
            rotationSymmetry = int(re.search("[0-9]y", symmetry).group()[0])
            rotated_hologram = self.rotate_hologram_euler_angle(0, 90, 90, False)
            rotated_hologram.symmetrize_hologram("z" + str(rotationSymmetry), True)
            rotated_hologram.rotate_hologram_euler_angle(0, -90, -90)
            Hologram_Temp = rotated_hologram.Hologram.copy()
        if "mz" in symmetry:
            SymmetrizedHologram = np.copy(Hologram_Temp)
            SymmetrizedHologram += np.flip(Hologram_Temp, axis=0).copy()
            Hologram_Temp = SymmetrizedHologram
            duplicate_symmetry *= 2
        if "mx" in symmetry:
            SymmetrizedHologram = np.copy(Hologram_Temp)
            SymmetrizedHologram += np.flip(Hologram_Temp, axis=1).copy()
            Hologram_Temp = SymmetrizedHologram
            duplicate_symmetry *= 2
        if "my" in symmetry:
            SymmetrizedHologram = np.copy(Hologram_Temp)
            SymmetrizedHologram += np.flip(
                np.roll(Hologram_Temp, phiSize // 2, axis=1).copy(), axis=1
            ).copy()
            Hologram_Temp = SymmetrizedHologram
            duplicate_symmetry *= 2
        Hologram_Temp /= duplicate_symmetry
        if is_update_self:
            self.Hologram = Hologram_Temp
            self.shape = self.Hologram.shape
        return_hologram = TyHologram()
        return_hologram.set_hologram_from_array(
            Hologram_Temp,
            save_path=self.directory,
            step_theta_deg=self.step_deg,
            step_phi_deg=self.step_deg,
        )
        return return_hologram

    def rotate_hologram_euler_angle(
        self,
        alpha_deg: float = 0,
        beta_deg: float = 0,
        gamma_deg: float = 0,
        is_update_self=False,
        is_calculate_high_resolution_hologram=False,
    ):
        alpha = np.deg2rad(alpha_deg)
        beta = np.deg2rad(beta_deg)
        gamma = np.deg2rad(gamma_deg)
        converted_hologram = np.zeros_like(self.Hologram)
        hologram_temp = self
        if is_calculate_high_resolution_hologram:
            hologram_temp = self.change_resolution_hologram(
                self.step_deg / 2, is_update_self=False
            )

        matrix_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        matrix_rotation = np.dot(
            np.array(
                [
                    [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1],
                ]
            ),
            matrix_rotation,
        )
        matrix_rotation = np.dot(
            np.array(
                [
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)],
                ]
            ),
            matrix_rotation,
        )
        matrix_rotation = np.dot(
            np.array(
                [
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1],
                ]
            ),
            matrix_rotation,
        )
        matrix_rotation = np.linalg.inv(matrix_rotation)
        for i in range(converted_hologram.shape[0]):
            theta_after_convert = self.theta_rad[i]
            for j in range(converted_hologram.shape[1]):
                phi_after_convert = self.phi_rad[j]
                r_hologram_after_convert = np.array(
                    [
                        np.sin(theta_after_convert) * np.cos(phi_after_convert),
                        np.sin(theta_after_convert) * np.sin(phi_after_convert),
                        np.cos(theta_after_convert),
                    ]
                )
                r_hologram_before_convert = np.dot(
                    matrix_rotation, r_hologram_after_convert
                )
                theta_before_convert = np.arccos(
                    min(max(r_hologram_before_convert[2], -1), 1)
                )
                phi_before_convert = np.arctan2(
                    r_hologram_before_convert[1], r_hologram_before_convert[0]
                )
                converted_hologram[i, j] = (
                    hologram_temp.get_intensity_from_angle_radian(
                        theta_before_convert, phi_before_convert
                    )
                )
        if is_update_self:
            self.Hologram = converted_hologram
            self.shape = self.Hologram.shape
        return_hologram = TyHologram()
        return_hologram.set_hologram_from_array(
            converted_hologram,
            save_path=self.directory,
            step_theta_deg=self.step_deg,
            step_phi_deg=self.step_deg,
        )
        return return_hologram


# %%
if __name__ == "__main__":
    holo = TyHologram("./holo.csv", energy=10000)
    # print(holo.Hologram)
    holo.rotate_hologram_euler_angle(0, 90, 0, is_update_self=True)
    holo.show2D()
