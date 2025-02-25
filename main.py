# 28062022 Моделирование концепций гравитации:
# Ньютон - притяжение (тривиальная задача).
# Ломоносов - отталкивание (приталкивание) с учетом того что есть остальная вселенная
# на макро уровне ковра галактик (пока без полостей вблизи массивных тел)
# 22222

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
import matplotlib.colors as mcolors


count = 0
trigger = False
G = 50000  # постоянная включающая в себя константы взаимодействия тел и пр.
np.random.seed(121)  # инициализируем датчик случайнхы чисел


# создать класс "Пробная вселенная "
class Trial_universe:
    def __init__(self, concept_sign=-1, G=100000, N=10, L=1000, n=3, Rm=50):
        self.concept_sign = concept_sign  # минус для приталкивания
        self.G = G  # постоянная включающая в себя константы взаимодействия тел и пр.
        self.N = N  # количество галактик (объектов) в тестовом кубе
        self.L = L  # размер тестового куба вселенной в относительных единицах длины
        self.n = n  # количество галактик на ребере тестового куба (для решетки)
        self.Rm = Rm  # максимальный размер галактики в относительных единицах (минимальный = 1). Размеры галактик отличаются на 1-2 порядка
        self.arr_inner_gal = self.inner_galaxies(N=self.N, L=self.L,
                                                 Rm=self.Rm)  # вызываем метод егнерации внутренних объектов при инициализации объекта
        self.cub = self.cub_of_galaxies(self.L, self.n)
        if self.concept_sign == -1:  # для концепции ломоносова генерируем внешнюю сетку галактик
            self.arr_grid = self.grid_of_galaxies(self.L, self.n)

        else:  # для концепции ньютона
            self.arr_grid = [0, 0, 0]
            self.number = 4

    def inner_galaxies(self, N, L, Rm=20):
        ### создаем случайные галактики внутри куба ###
        body = []
        for i in range(N):
            body.append([np.random.randint(1, L - 1), np.random.randint(1, L - 1), np.random.randint(1, L - 1),
                         np.random.randint(1, Rm), 0])
        dim_body = np.array(body)
        return dim_body

    def grid_of_galaxies(self, L, n, size=1):
        ### создаем неподвижную решетку внешних галактик по граням тестового куба ###
        l_min = L / n  # мин расстояние между галактиками на гранях тестового куба (для решетки)
        edge = []  # пустой список
        # формат данных для объекта [x,y,z, размер, признак 1-неподвижный 0-подвижный]
        # формируем грани
        for k in range(n + 1):
            for i in range(n + 1):
                for j in range(n + 1):
                    if k == 0 or k == n:
                        edge.append([i * l_min, j * l_min, k * l_min, size, 1])
                    if (0 < k < n) and (i == 0 or i == n or j == 0 or j == n):
                        edge.append([i * l_min, j * l_min, k * l_min, size, 1])
        dim_edge = np.array(edge)  # массив с гранями куба
        return dim_edge

    def cub_of_galaxies(self, L, n, size=5):
        ### создаем куб полностью равномерно заполненный галактиками ###
        l_min = L / n  # мин расстояние между галактиками на гранях тестового куба (для решетки)
        edge = []  # пустой список
        # формат данных для объекта [x,y,z, размер, признак 1-неподвижный 0-подвижный]
        # формируем грани
        for k in range(n + 1):
            for i in range(n + 1):
                for j in range(n + 1):
                    edge.append([i * l_min, j * l_min, k * l_min, size, 0])

        dim_edge = np.array(edge)  # массив
        return dim_edge

    def sphere_of_galaxies(self):
        ### сделать сферу галактик с равномерным распределением объектов на сфере
        # с постоянным потенциалом давления
        pass


class Vector_calc:
    def __init__(self, A, B, Mb):
        ### Рассчет сил взаимодействия - -смещения объекта А (нужна только масса объекта B) ###
        self.A = A
        self.B = B
        self.Mb = Mb
        self.vector = B - A
        self.sq_dist = sum(self.vector ** 2)  # расчет модуля (корень квадратный из суммы квадратов)
        self.dist_modul = np.sqrt(self.sq_dist)
        self.normalized_vector = self.vector / self.dist_modul  # нормируем - делим на модуль
        self.unit_interaction_force = G / self.sq_dist  # сила взаимодействия для единичных масс
        self.displacement_a = self.Mb * self.unit_interaction_force  # модуль смещение объекта A
        self.displcm_vector_a = self.normalized_vector * self.displacement_a  # вектор смещения объекта А


def force_calculation(dim_galaxies):
    ### расчет силового взаимодействия объектов друг с другом ###
    # на входе массив галактик, на выходе массив векторов смещения
    # триггер включения остановки
    s_max = 50  # расстояние на котором будет происходить поглощение объектов
    row, col = dim_galaxies.shape  # определяем количество строк

    dim_displacement = np.zeros((row, 3))  # массив для расчетного смещения
    for i in range(
            row):  # перебираем все точки массива и считаем суммарный вектор смещения для каждой с каждой (кроме самой себя)
        A = dim_galaxies[i, :3]
        Ma = dim_galaxies[i, 3]
        stationar_i = dim_galaxies[i, 4]  # извлекаем стационарность объекта =1
        dispacement = [0, 0, 0]
        for k in range(row):
            B = dim_galaxies[k, :3]  # срез для извлечения координат из массива
            Mb = dim_galaxies[k, 3]  # извлечение размера (массы) из массива
            stationar_k = dim_galaxies[k, 4]  # извлекаем стационарность объекта =1
            if k != i and Mb != 0 and Ma != 0:
                p = Vector_calc(A, B, Mb)

                # делаем проверку на то что объект подвижный
                if stationar_i == 1 and stationar_k == 1 and trigger:
                    dispacement = [0, 0, 0]  # неподвижные не двигаются т.е смещение равно 0
                else:
                    dispacement = dispacement + p.displcm_vector_a  # рассчитываем сумму векторов смещения
                # dispacement = dispacement + p.displcm_vector_a  # рассчитываем сумму векторов смещения

                # проверяем близость объектов
                s = p.dist_modul  # расстояние между точками A и B
                if s <= s_max:  # если расстояние маленькое (меньше чем s_max) то
                    if Mb <= Ma:  # больший поглощает меньший:
                        dim_galaxies[i, 3] = Mb + Ma
                        dim_galaxies[k, 3] = 0

                # проверяем что объект вышел за пределы куба - тогда он исчезает
                # if (np.max(p.B) > 1000 or np.min(p.B) < 0) and (stationar_i == 0 and stationar_k == 0):
                #№     dim_galaxies[k, 3] = 0

        dim_displacement[i] = dispacement  # запоминаем расчеты смещения в массиве
        # сделать нормирование и приведение к максимальному возможному скачку (s_max = например 1% от размера куба L)
        # для того чтобы избежать больших скачков.
    return (offset_normalization(dim_displacement)).astype(int, copy=False)


def convert_vector_to_modulus(arr_3d):
    # на входе массив трехмерных векторов, на выходе массив модулей
    row, col = arr_3d.shape
    modul = []
    for i in range(row):
        b = arr_3d[i, :3]
        mod = np.sqrt(sum(b ** 2))
        modul.append(mod)
    arr_modul = np.array(modul)
    return arr_modul


def offset_normalization(arr_displ, s_max=10):
    # на входе массив смещений, на выходе нормализованный массив смещений.
    # сделать нормирование и приведение к максимальному возможному скачку (s_max = например 1% от размера куба L)
    modul = convert_vector_to_modulus(arr_displ)
    max_value = np.max(modul)  # находим максимум в массиве модулей
    if max_value > s_max:
        k_norm = s_max / max_value  # находим коэффициент нормализации
    else:
        k_norm = 1
    return arr_displ * k_norm


# тестовая зона 2 (на объектах)
print('тестовая зона 2 (на объектах)')
print('ломоносов')
lomonosov_universe = Trial_universe(N=1)
print('внутренние галактики')
print(lomonosov_universe.arr_inner_gal)
print('решетка')
print(lomonosov_universe.arr_grid)

print('ньютон')
newtons_universe = Trial_universe(1, N=10)
print('внутренние галактики')
print(newtons_universe.arr_inner_gal)


def uptd(i):
    global count
    global trigger
    count += 1
    # print(count)
    if count == 1:
        trigger = True

    # объявляем глобальными массивы чтобы можно было их изменять из функции
    global newton
    global lom
    # рассчитываем смещение
    disp_newton = force_calculation(newton)
    dx1 = disp_newton[:, 0]
    dy1 = disp_newton[:, 1]
    dz1 = disp_newton[:, 2]
    disp_lom = force_calculation(lom)
    dx2 = disp_lom[:, 0]
    dy2 = disp_lom[:, 1]
    dz2 = disp_lom[:, 2]
    # прибавляем или вычитаем к абсолютному положению = новое положение галактик

    newton[:, [0, 1, 2]] = newton[:, [0, 1, 2]] + disp_newton # притяжение
    x1 = newton[:, 0]
    y1 = newton[:, 1]
    z1 = newton[:, 2]
    s1 = newton[:, 3]
    lom[:, [0, 1, 2]] = lom[:, [0, 1, 2]] - disp_lom # отталкивание
    x2 = lom[:, 0]
    y2 = lom[:, 1]
    z2 = lom[:, 2]
    s2 = lom[:, 3]
    label = lom[:, 4]
    # рисуем:
    # print("clear")
    ax1.clear()  # очистка графика
    ax2.clear()
    ax1.set_title('newton')
    ax2.set_title('lomonosov')
    # выводим рещетку на обеих графиках
    ax1.scatter(lom_g[:, 0], lom_g[:, 1], lom_g[:, 2], s=lom_g[:, 3], c='g', alpha=0)
    # ax2.scatter(lom_g[:, 0], lom_g[:, 1], lom_g[:, 2], s=lom_g[:, 3], c='g')
    # выводим положение галактик
    ax1.scatter(x1, y1, z1, s=s1, c='b', alpha=0.5)
    # ax2.scatter(x2, y2, z2, s=s2, c='r')
    ax2.scatter(x2, y2, z2, s=s2, c=label, cmap=mcolors.ListedColormap(["r", "b", "g", "y", "c", "k"]))
    # выводим стрелки
    ax1.quiver(x1, y1, z1, dx1, dy1, dz1, length=30, pivot='tail')
    ax2.quiver(x2, y2, z2, -dx2, -dy2, -dz2, length=30, pivot='tail')


# диаграммы
fig1 = plt.figure()
ax1 = fig1.add_subplot(121, projection='3d')
ax2 = fig1.add_subplot(122, projection='3d')
ax1.set_title('newton')
ax2.set_title('lomonosov')
# делаем полные копии массивов
# newton = newtons_universe.arr_inner_gal.copy()
# newton = lomonosov_universe.arr_inner_gal.copy()  # делаем одинаковые исходные данные для 2-х моделей
# lom = lomonosov_universe.arr_inner_gal.copy()  # делаем одинаковые исходные данные для 2-х моделей
###lom_g = lomonosov_universe.arr_grid.copy() # решетка
lom_g = lomonosov_universe.cub # заполненный куб
lom = np.vstack([lomonosov_universe.arr_inner_gal, lom_g])  # объединенный массив внутренних объектов и решетки

# lom[0, :] = [500, 500, 500, 100, 1]  # заменим на объект с нужными параметрами (большой объект посредине куба)
# lom[1, :] = [520, 500, 500, 5, 0]
# lom[2, :] = [520, 540, 500, 1, 0]
# lom[3, :] = [520, 540, 520, 1, 0]

newton = lom.copy()
# print(lom)
# выводим рещетку на обеих графиках
ax1.scatter(lom_g[:, 0], lom_g[:, 1], lom_g[:, 2], s=lom_g[:, 3], c='b', alpha=0)
# ax2.scatter(lom_g[:, 0], lom_g[:, 1], lom_g[:, 2], s=lom_g[:, 3], c='g')
# выводим положение галактик
ax1.scatter(newton[:, 0], newton[:, 1], newton[:, 2], s=newton[:, 3], c='b')
ax2.scatter(lom[:, 0], lom[:, 1], lom[:, 2], s=lom[:, 3], c='r')
# анимация
ani1 = matplotlib.animation.FuncAnimation(fig1, uptd, frames=20, interval=100)
plt.show()

# newtons_universe = Trial_universe(1, N=12)
