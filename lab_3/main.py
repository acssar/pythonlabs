class Worker:

    __SALARY = 0
    __TASKS_DONE = 0
    _JOB_SIGN = 1

    def __init__(self):
        self.__money = self.__SALARY
        self.__tasks = self.__TASKS_DONE
        self.__sign = self._JOB_SIGN

    @property
    def salary(self):
        return self.__money

    @salary.setter
    def salary(self, sal):
        self.__money += sal * self.__tasks
        self.__tasks = 0

    @classmethod
    def set_prepaid_salary(cls, sal=0):
        cls.__SALARY = sal

    def do_work(self, matrix1, matrix2):
        """ Function to do matrix operations depending of class parameter "sign"
            it either add or sub two matrix by element.

        Parameters
        ----------
        matrix1: str
            Name of the file with 1st matrix.
        matrix2: str
            Name of the file with 2nd matrix.

        Returns
        -------
        None

        Prints
        ------
        Matrix result

        """
        with open(matrix1, 'r') as m1:
            a = [[int(num) for num in line.split(' ')] for line in m1]
        with open(matrix2, 'r') as m2:
            b = [[int(num) for num in line.split(' ')] for line in m2]
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise AttributeError("matrix should have same dims")
        res = []
        arr = []
        for i in range(len(a)):
            for j in range(len(a[0])):
                arr.append(a[i][j] + self.__sign * b[i][j])
            res.append(arr)
            arr = []
        self.__tasks += 1
        for row in res:
            print(' '.join(list(map(str, row))))


class Lupa(Worker):

    _JOB_SIGN = -1

    def __init__(self):
        self.worker_type = "Lupa"
        super().__init__()


class Pupa(Worker):

    _JOB_SIGN = 1

    def __init__(self):
        self.worker_type = "Pupa"
        super().__init__()


class Accountant:
    __LUPA_REGIONAL_COEFFICIENT = 1.33
    __PUPA_REGIONAL_COEFFICIENT = 1.1
    __STANDART_SALARY = 10

    def __init__(self):
        self.__lupa_coeff = self.__LUPA_REGIONAL_COEFFICIENT
        self.__pupa_coeff = self.__PUPA_REGIONAL_COEFFICIENT
        self.__salary = self.__STANDART_SALARY

    def give_salary(self, worker):
        """Function to give salary to workers depending on worker type, regional coefficient and tasks done by worker.

        Parameters
        ----------
        worker: class Worker(Lupa or Pupa) object

        """
        if isinstance(worker, Lupa):
            worker.salary = self.__salary * self.__lupa_coeff
        elif isinstance(worker, Pupa):
            worker.salary = self.__salary * self.__pupa_coeff
        else:
            raise print("Worker type is unknown")


if __name__ == '__main__':
    Lupa.set_prepaid_salary(10#)

    worker1 = Lupa()
    worker2 = Pupa()
    worker1.do_work("matrix1.txt", "matrix2.txt")
    worker1.do_work("matrix3.txt", "matrix4.txt")
    worker2.do_work("matrix1.txt", "matrix2.txt")
    print("worker 1 - " + worker1.worker_type + "'s money = " + str(worker1.salary))
    print("worker 2 - " + worker2.worker_type + "'s money = " + str(worker2.salary))
    acc = Accountant()
    acc.give_salary(worker1)
    print("worker 1 - " + worker1.worker_type + "'s money = " + str(worker1.salary))
    acc.give_salary(worker1)
    print("worker 1 - " + worker1.worker_type + "'s money = " + str(worker1.salary))
    acc.give_salary(worker2)
    print("worker 2 - " + worker2.worker_type + "'s money = " + str(worker2.salary))
