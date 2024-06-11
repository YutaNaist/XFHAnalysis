import random
import numpy as np
from typing import List


class TyCrystal:
    number_of_atom = 0
    crystal_vector = [0, 0, 0, 0, 0, 0]
    atom_names = [["AA"]]
    atom_positions_crystal = [[0, 0, 0]]
    atom_positions_xyz = [[0, 0, 0]]
    matrix = np.zeros((3, 3))

    def __init__(self, xtlFileName: str = ""):
        if xtlFileName != "":
            self.loadFromXtlFile(xtlFileName)

    def loadFromXtlFile(self, FileName):
        # FileDirectory = ""
        list_atom_names = []
        list_atom_positions_crystal = []
        list_atom_positions_xyz = []
        crystal_vector = []
        with open(FileName, mode="r") as File:
            LinesFile = File.read().split("\n")
            flagRead = 0
            for i in range(len(LinesFile)):
                OneLine = LinesFile[i].split(" ")
                if i == 2:
                    for j in range(len(OneLine)):
                        if OneLine[j] == "":
                            continue
                        else:
                            crystal_vector.append(float(OneLine[j]))
                    self.crystal_vector = crystal_vector
                    self.create_matrix_to_xyz()
                    # print(self.crystal_vector)
                    # print(self.matrix)
                    # print()
                elif OneLine[0] == "NAME":
                    flagRead = 1
                elif OneLine[0] == "EOF":
                    flagRead = 0
                elif flagRead == 1:
                    Coordinate = []
                    Atom = ""
                    for j in range(len(OneLine)):
                        if j == 0:
                            Atom = OneLine[j]
                        elif OneLine[j] == "":
                            continue
                        else:
                            Coordinate.append(float(OneLine[j]))
                    if (
                        0 <= Coordinate[0] < 1
                        and 0 <= Coordinate[1] < 1
                        and Coordinate[2] < 1
                    ):
                        list_atom_positions_crystal.append(Coordinate)
                        list_atom_positions_xyz.append(
                            self.crystalCoordinateToXYZCoordinate(Coordinate)
                        )
                        # print()
                        list_atom_names.append(Atom)
        self.atom_names = list_atom_names
        self.atom_positions_crystal = list_atom_positions_crystal
        self.atom_positions_xyz = list_atom_positions_xyz
        self.number_of_atom = len(list_atom_names)
        # self.create_matrix_to_xyz()
        return crystal_vector, list_atom_names, list_atom_positions_crystal

    def create_matrix_to_xyz(self):
        self.matrix[2] = np.array((0, 0, self.crystal_vector[2]))
        self.matrix[0] = np.array(
            [
                self.crystal_vector[0] * np.sin(np.radians(self.crystal_vector[4])),
                0,
                self.crystal_vector[0] * np.cos(np.radians(self.crystal_vector[4])),
            ]
        )
        bz = np.cos(np.radians(self.crystal_vector[3]))
        bx = (
            np.cos(np.radians(self.crystal_vector[5]))
            - np.cos(np.radians(self.crystal_vector[3]))
            * np.cos(np.radians(self.crystal_vector[4]))
        ) / np.sin(np.radians(self.crystal_vector[4]))
        by = np.sqrt(1 - bz**2 - bx**2)
        self.matrix[1] = np.array((bx, by, bz)) * self.crystal_vector[1]
        # lena = np.sqrt(np.sum((self.matrix[0] + self.matrix[1])**2))
        # lenb = np.sqrt(np.sum((self.matrix[1] + self.matrix[2])**2))
        # lenc = np.sqrt(np.sum((self.matrix[0] + self.matrix[2])**2))
        # lena = np.sqrt(np.sum((self.matrix[0]) ** 2))
        # lenb = np.sqrt(np.sum((self.matrix[1]) ** 2))
        # lenc = np.sqrt(np.sum((self.matrix[2]) ** 2))
        # print(lena)
        # print(lenb)
        # print(lenc)
        # print(np.dot(self.matrix[0],self.matrix[1])/ lena/lenb)
        # print(np.dot(self.matrix[1],self.matrix[2])/ lenb/lenc)
        # print(np.dot(self.matrix[0],self.matrix[2])/ lena/lenc)
        return self.matrix

    def crystalCoordinateToXYZCoordinate(self, crystal_coordinate):
        # print(crystal_coordinate)
        matrix = self.matrix
        # crystal_coordinate = np.array(crystal_coordinate)
        xyz_coordinate = np.dot(matrix, crystal_coordinate)
        # print(xyz_coordinate, np.sqrt(np.sum(xyz_coordinate**2)))
        # print(matrix)
        return xyz_coordinate.tolist()

    def xyzCoordinateToCrystalCoordinate(self, xyz_coordinate):
        matrix = self.matrix
        xyz_coordinate = np.array(xyz_coordinate)
        crystal_coordinate = np.dot(np.linalg.inv(matrix), xyz_coordinate)
        return crystal_coordinate.tolist()

    def writeToXtlFile(self, FileName):
        with open(FileName, mode="w") as File:
            File.write("TITLE\n")
            File.write("CELL\n")
            for i in range(len(self.crystal_vector)):
                File.write(" {}".format(self.crystal_vector[i]))
            File.write("\n")
            File.write("SYMMETRY NUMBER 1\n")
            File.write("SYMMETRY LABEL  P1\n")
            File.write("ATOMS\n")
            File.write("NAME X Y Z\n")
            for i in range(self.number_of_atom):
                File.write(self.atom_names[i])
                for j in range(len(self.atom_positions_crystal[i])):
                    File.write(" {}".format(self.atom_positions_crystal[i][j]))
                File.write("\n")
            File.write("EOF\n")
        return FileName

    def changePositionInCrystalCoordinate(self, index, difPosition):
        if not (index < self.number_of_atom):
            return 1
        for i in range(len(difPosition)):
            # crystalCoordinate =
            self.atom_positions_crystal[index][i] += difPosition[i]
            if self.atom_positions_crystal[index][i] < 0:
                self.atom_positions_crystal[index][i] += 1
            elif self.atom_positions_crystal[index][i] >= 1:
                self.atom_positions_crystal[index][i] -= 1
        self.atom_positions_xyz[index] = self.crystalCoordinateToXYZCoordinate(
            self.atom_positions_crystal[index]
        )
        return 0

    def changePositionInCrystalCoordinateAngstrom(self, index, difPosition):
        if not (index < self.number_of_atom) or len(difPosition) != 3:
            return 1
        for i in range(len(difPosition)):
            self.atom_positions_crystal[index][i] += (
                difPosition[i] / self.crystal_vector[i]
            )
            if self.atom_positions_crystal[index][i] < 0:
                self.atom_positions_crystal[index][i] += 1
            elif self.atom_positions_crystal[index][i] >= 1:
                self.atom_positions_crystal[index][i] -= 1
        self.atom_positions_xyz[index] = self.crystalCoordinateToXYZCoordinate(
            self.atom_positions_crystal[index]
        )
        return 0

    def changePositionInXYZAngstrom(self, index, difPosition):
        if not (index < self.number_of_atom) or len(difPosition) != 3:
            return 1
        for i in range(len(difPosition)):
            self.atom_positions_xyz[index][i] += difPosition[i]
        self.atom_positions_crystal[index] = self.xyzCoordinateToCrystalCoordinate(
            self.atom_positions_xyz[index]
        )
        flag_change = 0
        for i in range(len(difPosition)):
            if self.atom_positions_crystal[index][i] < 0:
                self.atom_positions_crystal[index][i] += 1
                flag_change = 1
            elif self.atom_positions_crystal[index][i] >= 1:
                self.atom_positions_crystal[index][i] -= 1
                flag_change = 1
        if flag_change == 1:
            self.atom_positions_crystal[index] = self.xyzCoordinateToCrystalCoordinate(
                self.atom_positions_xyz[index]
            )
        return 0

    def expandLattice(self, a, b, c):
        self.crystal_vector[0] *= a
        self.crystal_vector[1] *= b
        self.crystal_vector[2] *= c
        for i in range(self.number_of_atom):
            self.atom_positions_crystal[i][0] = self.AtomPosition[i][0] / a
            self.atom_positions_crystal[i][1] = self.AtomPosition[i][1] / b
            self.atom_positions_crystal[i][2] = self.AtomPosition[i][2] / c
        new_atom_positions_crystal = []
        new_atom_positions_xyz = []
        new_atom_names = []
        new_number_of_atom = 0
        for ic in range(int(c)):
            for ib in range(int(b)):
                for ia in range(int(a)):
                    for i in range(self.number_of_atom):
                        atom_position_crystal = [
                            self.atom_positions_crystal[i][0] + ia / a,
                            self.atom_positions_crystal[i][1] + ib / b,
                            self.atom_positions_crystal[i][2] + ic / c,
                        ]
                        new_atom_positions_crystal.append(atom_position_crystal)
                        new_atom_positions_xyz.append(
                            self.crystalCoordinateToXYZCoordinate(atom_position_crystal)
                        )
                        new_atom_names.append(self.AtomName[i])
                        new_number_of_atom += 1
        self.number_of_atom = new_number_of_atom
        self.atom_positions_crystal = new_atom_positions_crystal
        self.atom_positions_xyz = new_atom_positions_xyz
        self.atom_names = new_atom_names
        return 0

    def offsetAtoms(self, a, b, c):
        for i in range(self.number_of_atom):
            self.atom_positions_crystal[i][0] += a
            self.atom_positions_crystal[i][1] += b
            self.atom_positions_crystal[i][2] += c
            self.atom_positions_xyz[i] = self.crystalCoordinateToXYZCoordinate(
                self.atom_positions_crystal[i]
            )

    def changeAtom(self, index, AtomicName):
        if not (index < self.number_of_atom):
            return 1
        self.atom_names[index] = AtomicName
        return 0

    def randomSubstitute(self, AtomicNameBefore, AtomicNameAfter, fluctuation):
        for i in range(self.number_of_atom):
            if self.atom_names[i] == AtomicNameBefore and random.random() < fluctuation:
                self.atom_names[i] = AtomicNameAfter
        return 0

    def randomDisplacementAtom(self, AtomicName, StandardDeviationAngstrom):
        StandardDeviationAngstrom /= np.sqrt(3)
        for i in range(self.number_of_atom):
            if self.atom_names[i] == AtomicName:
                for j in range(3):
                    self.atom_positions_crystal[i][j] += (
                        random.normalvariate(0, StandardDeviationAngstrom)
                        / self.crystal_vector[j]
                    )
                self.atom_positions_xyz[i] = self.crystalCoordinateToXYZCoordinate(
                    self.atom_positions_crystal[i]
                )
        return 0

    def randomDisplacementIndex(self, index, StandardDeviationAngstrom):
        StandardDeviationAngstrom /= np.sqrt(3)
        for j in range(3):
            self.atom_positions_crystal[index][j] += (
                random.normalvariate(0, StandardDeviationAngstrom)
                / self.crystal_vector[j]
            )
        self.atom_positions_xyz[index] = self.crystalCoordinateToXYZCoordinate(
            self.atom_positions_crystal[index]
        )
        return 0

    def randomDisplacementAlongCrystalVectorDirection(
        self, AtomicName, StandardDeviation, Direction
    ):
        Norm = np.sqrt(Direction[0] ** 2 + Direction[1] ** 2 + Direction[2] ** 2)
        NormDirection = [Direction[0] / Norm, Direction[1] / Norm, Direction[2] / Norm]
        for i in range(self.number_of_atom):
            if self.atom_names[i] == AtomicName:
                rand = random.normalvariate(0, StandardDeviation)
                for j in range(3):
                    self.atom_positions_crystal[i][j] += (
                        NormDirection[j] * rand / self.crystal_vector[j]
                    )
                self.atom_positions_xyz[i] = self.crystalCoordinateToXYZCoordinate(
                    self.atom_positions_crystal[i]
                )
        return 0

    def randomDisplacementAlongCrystalVectorDirectionIndex(
        self, index, StandardDeviation, Direction
    ):
        Norm = np.sqrt(Direction[0] ** 2 + Direction[1] ** 2 + Direction[2] ** 2)
        NormDirection = [Direction[0] / Norm, Direction[1] / Norm, Direction[2] / Norm]
        rand = random.normalvariate(0, StandardDeviation)
        for j in range(3):
            self.atom_positions_crystal[index][j] += (
                NormDirection[j] * rand / self.crystal_vector[j]
            )
        self.atom_positions_xyz[index] = self.crystalCoordinateToXYZCoordinate(
            self.atom_positions_crystal[index]
        )
        return 0

    def changeLatticeParameter(self, a=0, b=0, c=0, alpha=0, beta=0, gamma=0):
        if a != 0:
            self.crystal_vector[0] = a
        if b != 0:
            self.crystal_vector[1] = b
        if c != 0:
            self.crystal_vector[2] = c
        if alpha != 0:
            self.crystal_vector[3] = alpha
        if beta != 0:
            self.crystal_vector[4] = beta
        if gamma != 0:
            self.crystal_vector[5] = gamma
        self.create_matrix_to_xyz()
        for i in range(self.number_of_atom):
            self.atom_positions_xyz[i] = self.crystalCoordinateToXYZCoordinate(
                self.atom_positions_crystal[i]
            )
        return self.crystal_vector

    def showAtoms(self):
        for i in range(self.number_of_atom):
            print(
                "{} - {}: {}, {}, {}".format(
                    i,
                    self.atom_names[i],
                    self.atom_positions_crystal[i][0],
                    self.atom_positions_crystal[i][1],
                    self.atom_positions_crystal[i][2],
                )
            )
        return 0

    def showAtomsAbsolute(self):
        for i in range(self.number_of_atom):
            print(
                "{} - {}: {}, {}, {}".format(
                    i,
                    self.atom_names[i],
                    self.atom_positions_xyz[i][0],
                    self.atom_positions_xyz[i][1],
                    self.atom_positions_xyz[i][2],
                )
            )
        return 0

    def numberOfAtom(self, AtomicName):
        count = 0
        for i in range(self.numberOfAtom):
            if self.atom_names[i] == AtomicName:
                count += 1
        return count

    def createClusterCsv(
        self,
        emitter: str,
        cluster_size: float,
        direction_Z: List[int] = [0, 0, 1],
        direction_Y: List[int] = [0, 1, 0],
        filename_base: str = "Cluster",
        is_include_emitter: bool = False,
        option="",
    ):
        DirectionZ = np.array(direction_Z)
        DirectionY = np.array(direction_Y)
        DirectionX = np.cross(DirectionY, DirectionZ)
        DirectionMatrix = np.array(
            [
                DirectionX / np.linalg.norm(DirectionX, ord=2),
                DirectionY / np.linalg.norm(DirectionY, ord=2),
                DirectionZ / np.linalg.norm(DirectionZ, ord=2),
            ]
        )
        DirectionMatrixInv = np.linalg.inv(DirectionMatrix)
        CoordinateInXYZ = self.matrix

        # repeatMax = int(cluster_size / min(
        #     [self.crystal_vector[0], self.crystal_vector[1], self.crystal_vector[2]]))
        repeat_max_x = int(cluster_size / self.crystal_vector[0])
        repeat_max_y = int(cluster_size / self.crystal_vector[1])
        repeat_max_z = int(cluster_size / self.crystal_vector[2])

        list_all_atoms = []
        number_of_all_atoms = []
        emitter_No = 0
        for i in range(self.number_of_atom):
            atom_name = self.atom_names[i]
            if atom_name == emitter:
                origin_position = np.array(self.atom_positions_xyz[i])
                with open(
                    "{}_{}.csv".format(filename_base, emitter_No), mode="w"
                ) as file:
                    count_atom = 0
                    for j in range(self.number_of_atom):
                        scatter_name = self.atom_names[j]
                        scatter_position_base_xyz = (
                            np.array(self.atom_positions_xyz[j]) - origin_position
                        )
                        for ix in range(-repeat_max_x - 1, repeat_max_x + 2):
                            for iy in range(-repeat_max_y - 1, repeat_max_y + 2):
                                for iz in range(-repeat_max_z - 1, repeat_max_z + 2):
                                    index_vector = np.array([ix, iy, iz])
                                    position = scatter_position_base_xyz + np.dot(
                                        CoordinateInXYZ, index_vector
                                    )
                                    r = np.linalg.norm(position, ord=2)
                                    if r > cluster_size:
                                        continue
                                    if r == 0 and is_include_emitter == False:
                                        continue
                                    position = np.dot(position, DirectionMatrixInv)
                                    file.write(
                                        "{0},{1},{2},{3}\n".format(
                                            scatter_name,
                                            position[0],
                                            position[1],
                                            position[2],
                                        )
                                    )
                                    count_atom += 1
                emitter_No += 1

    def createClusterXYZ(
        self,
        emitter: str,
        clusterSize: float,
        directionZ: List[int] = [0, 0, 1],
        directionY: List[int] = [0, 1, 0],
        filenameBase: str = "Cluster",
        outputDirectory: str = "./",
        isIncludeEmitter: bool = False,
        comment: str = "",
        option="",
    ):
        DirectionZ = np.array(directionZ)
        DirectionY = np.array(directionY)
        DirectionX = np.cross(DirectionY, DirectionZ)
        DirectionMatrix = np.array(
            [
                DirectionX / np.linalg.norm(DirectionX, ord=2),
                DirectionY / np.linalg.norm(DirectionY, ord=2),
                DirectionZ / np.linalg.norm(DirectionZ, ord=2),
            ]
        )
        DirectionMatrixInv = np.linalg.inv(DirectionMatrix)
        CoordinateInXYZ = self.matrix

        # repeatMax = int(cluster_size / min(
        #     [self.crystal_vector[0], self.crystal_vector[1], self.crystal_vector[2]]))
        repeat_max_x = int(clusterSize / self.crystal_vector[0])
        repeat_max_y = int(clusterSize / self.crystal_vector[1])
        repeat_max_z = int(clusterSize / self.crystal_vector[2])

        number_of_all_atoms = []
        emitter_No = 0
        listAllClusters = []
        for i in range(self.number_of_atom):
            atom_name = self.atom_names[i]
            if atom_name == emitter:
                print("\rEmitter: {}".format(emitter_No), end="")
                origin_position = np.array(self.atom_positions_crystal[i])
                # with open(
                #     "{}_{}.xyz".format(filenameBase, emitter_No), mode="w"
                # ) as file:
                count_atom = 0
                list_all_atoms = []
                for j in range(self.number_of_atom):
                    scatter_name = self.atom_names[j]
                    scatter_position_base = (
                        np.array(self.atom_positions_crystal[j]) - origin_position
                    )
                    for ix in range(-repeat_max_x - 1, repeat_max_x + 2):
                        for iy in range(-repeat_max_y - 1, repeat_max_y + 2):
                            for iz in range(-repeat_max_z - 1, repeat_max_z + 2):
                                index_vector = np.array([ix, iy, iz])
                                position_crystal = scatter_position_base + index_vector
                                position = np.dot(CoordinateInXYZ, position_crystal)
                                # position = position_xyz
                                position = self.crystalCoordinateToXYZCoordinate(
                                    position_crystal
                                )
                                r = np.linalg.norm(position, ord=2)
                                if r > clusterSize:
                                    continue
                                if r == 0 and isIncludeEmitter == False:
                                    continue
                                # print(position, r, ix, iy, iz, np.rad2deg(np.arccos(position[2]/r)))
                                # print(
                                #     position,
                                #     r,
                                #     np.rad2deg(np.arccos(position[2] / r)),
                                #     np.rad2deg(np.arccos(position[0] / r)),
                                # )
                                position = np.dot(position, DirectionMatrixInv)
                                list_all_atoms.append(
                                    {
                                        "name": scatter_name,
                                        "x": position[0],
                                        "y": position[1],
                                        "z": position[2],
                                        "r": r,
                                    }
                                )
                                # file.write(
                                #     "{0}\t{1}\t{2}\t{3}\n".format(
                                #         scatter_name,
                                #         position[0],
                                #         position[1],
                                #         position[2],
                                #     )
                                # )
                                count_atom += 1
                # file_str = ""
                list_all_atoms = sorted(list_all_atoms, key=lambda x: x["r"])
                # with open(
                #     "{}_{}.xyz".format(filenameBase, emitter_No), mode="r"
                # ) as file:
                #     file_str = file.read()
                with open(
                    "{}{}_{}.xyz".format(outputDirectory, filenameBase, emitter_No),
                    mode="w",
                ) as file:
                    file.write("{0}\n".format(count_atom))
                    if comment != "":
                        file.write("{0}\n".format(comment))
                    else:
                        file.write("Generated by Python Code\n")
                    # file.write(file_str)
                    for i in range(len(list_all_atoms)):
                        file.write(
                            "{0}\t{1}\t{2}\t{3}\n".format(
                                list_all_atoms[i]["name"],
                                list_all_atoms[i]["x"],
                                list_all_atoms[i]["y"],
                                list_all_atoms[i]["z"],
                            )
                        )
                listAllClusters.append(list_all_atoms)
                emitter_No += 1
        print()
        return listAllClusters
        # position_base =
        # print()

    def createClusterListDict(
        self,
        emitter: str,
        clusterSize: float,
        directionZ: List[int] = [0, 0, 1],
        directionY: List[int] = [0, 1, 0],
        isIncludeEmitter: bool = False,
    ):
        DirectionZ = np.array(directionZ)
        DirectionY = np.array(directionY)
        DirectionX = np.cross(DirectionY, DirectionZ)
        DirectionMatrix = np.array(
            [
                DirectionX / np.linalg.norm(DirectionX, ord=2),
                DirectionY / np.linalg.norm(DirectionY, ord=2),
                DirectionZ / np.linalg.norm(DirectionZ, ord=2),
            ]
        )
        DirectionMatrixInv = np.linalg.inv(DirectionMatrix)
        CoordinateInXYZ = self.matrix

        # repeatMax = int(cluster_size / min(
        #     [self.crystal_vector[0], self.crystal_vector[1], self.crystal_vector[2]]))
        repeat_max_x = int(clusterSize / self.crystal_vector[0])
        repeat_max_y = int(clusterSize / self.crystal_vector[1])
        repeat_max_z = int(clusterSize / self.crystal_vector[2])

        number_of_all_atoms = []
        emitter_No = 0
        listAllClusters = []
        for i in range(self.number_of_atom):
            atom_name = self.atom_names[i]
            if atom_name == emitter:
                print("\rEmitter: {}".format(emitter_No), end="")
                origin_position = np.array(self.atom_positions_crystal[i])
                # with open(
                #     "{}_{}.xyz".format(filenameBase, emitter_No), mode="w"
                # ) as file:
                count_atom = 0
                list_all_atoms = []
                for j in range(self.number_of_atom):
                    scatter_name = self.atom_names[j]
                    scatter_position_base = (
                        np.array(self.atom_positions_crystal[j]) - origin_position
                    )
                    for ix in range(-repeat_max_x - 1, repeat_max_x + 2):
                        for iy in range(-repeat_max_y - 1, repeat_max_y + 2):
                            for iz in range(-repeat_max_z - 1, repeat_max_z + 2):
                                index_vector = np.array([ix, iy, iz])
                                position_crystal = scatter_position_base + index_vector
                                position = np.dot(CoordinateInXYZ, position_crystal)
                                # position = position_xyz
                                position = self.crystalCoordinateToXYZCoordinate(
                                    position_crystal
                                )
                                r = np.linalg.norm(position, ord=2)
                                if r > clusterSize:
                                    continue
                                if r == 0 and isIncludeEmitter == False:
                                    continue
                                # print(position, r, ix, iy, iz, np.rad2deg(np.arccos(position[2]/r)))
                                # print(
                                #     position,
                                #     r,
                                #     np.rad2deg(np.arccos(position[2] / r)),
                                #     np.rad2deg(np.arccos(position[0] / r)),
                                # )
                                position = np.dot(position, DirectionMatrixInv)
                                list_all_atoms.append(
                                    {
                                        "name": scatter_name,
                                        "x": position[0],
                                        "y": position[1],
                                        "z": position[2],
                                        "r": r,
                                    }
                                )
                                # file.write(
                                #     "{0}\t{1}\t{2}\t{3}\n".format(
                                #         scatter_name,
                                #         position[0],
                                #         position[1],
                                #         position[2],
                                #     )
                                # )
                                count_atom += 1
                # file_str = ""
                # print(list_all_atoms)
                list_all_atoms = sorted(list_all_atoms, key=lambda x: x["r"])
                listAllClusters.append(list_all_atoms)
                emitter_No += 1
        print()
        return listAllClusters


if __name__ == "__main__":
    XTLFile = TyCrystal()()
    # XTLFile.readFromXtlFile("GaN_6m.xtl")
    # XTLFile.readFromXtlFile("Mg.xtl")
    # # print()
    # XTLFile.create_cluster_XYZ("Mg", 4)
    # XTLFile.create_cluster_XYZ("Ga", 10, is_include_emitter=True)
    # XTLFile.readFromXtlFile(
    #     "/Users/yuta/Desktop/ImageShift/CuVer2/Calculation/Cu.xtl")
    # # XTLFile.changePositionInLattice(3, [0.5, 0, 0])
    # # XTLFile.changePositionAbsoluteValue(3, [0, 0, 0.2])
    # XTLFile.expandLatice(2, 2, 2)
    # XTLFile.randomSubstitute("Cu", "Al", 0.5)
    # XTLFile.randomDisplacementAlongAtom("Al", 0.2, [1, 1, 1])
    # XTLFile.writeToXtlFile(
    #     "/Users/yuta/Desktop/ImageShift/CuVer2/Calculation/Cu001.xtl")
    # XTLFile.readFromXtlFile("/Users/yuta/Downloads/BTO.xtl")
    # XTLFile.expandLattice(2, 2, 2)
    # XTLFile.showAtoms()
    # XTLFile.changePositionAbsoluteValue(1, [0.1, 0, 0])
    # XTLFile.writeToXtlFile("/Users/yuta/Downloads/BTO3.xtl")

    XTLFile.readFromXtlFile("Mg.xtl")
    XTLFile.create_cluster_XYZ("Mg", 4)
