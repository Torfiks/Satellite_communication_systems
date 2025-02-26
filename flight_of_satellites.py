"""
Developed by Meshkov Roman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Satellite:
    def __init__(self, mass_object, speed_object, position_object, color="b"):
        self.mass = mass_object
        self.speed = speed_object
        self.position = position_object
        self.color = color
        self.pos = None
        self.positions_history = []  # Список для хранения истории позиций
        self.start_position = []
        self.status_position = False
        self.status_position_break = True

    def plot_sat(self, **kwargs):
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        r = self.mass * 0.2

        self.color =  kwargs["c"] if len(kwargs) != 0 else self.color
        self.pos = ax.scatter(x, y, z, **kwargs)


    def new_condition(self, num_iterations=50):
        """
        Функция new_condition предназначена для генерации координат на плоскости с заданной скорость.

        :param num_iterations: Определяет как быстро будет проходить анимация объекта,
        чем больше значение тем быстрее будет передвигать объект.

        В случе когда объект достигает своей первоначальной координаты данные в массив перестают добавляться,
        что позволяет не увеличивать нагрузку.
        """
        for _ in range(num_iterations):
            new_c = calculation(self.speed, self.position)
            self.speed = new_c[0]
            self.position = new_c[1]



        if len(self.positions_history) > 2:
            arrays = np.round([self.positions_history[-1][0], self.positions_history[-1][1], self.positions_history[-1][-1]], 1).tolist()
            if arrays == self.start_position:
                self.status_position_break = False
                # self.positions_history.clear()

        if self.status_position_break:
            self.positions_history.append(self.position.copy())  # Сохраняем текущее положение

    def plot_trail(self):
        """
        Функция отрисовки линии после объекта
        В случе когда объект достигает своей первоначальной координаты, линия не рисуется дальше,
        что позволяет не увеличивать нагрузку.
        """

        if len(self.positions_history) < 1:
            return

        xs = [pos[0] for pos in self.positions_history]
        ys = [pos[1] for pos in self.positions_history]
        zs = [pos[2] for pos in self.positions_history]

        array = [round(xs[-1],1), round(ys[-1],1), round(zs[-1],1)]

        if not self.status_position and len(self.start_position) == 0:
            self.status_position = True
            self.start_position = array

        ax.plot(xs, ys, zs, color=self.color, alpha=0.5)  # Рисуем линию траектории

dt = 0.0005  # минимальный квант времени dt

def calculation(speed_object, position_object):
    """
    Функция рассчитывающая координаты объекта
    :param speed_object: скорость объекта
    :param position_object: положение объекта
    :return:
    """
    sum_quadratic = np.sum(position_object ** 2)  # сумма квадратов
    vector = position_object / np.sqrt(sum_quadratic ** 3)  # вектор ускорения
    speed_object += -vector * dt  # скорость (со знаком МИНУС)
    position_object += speed_object * dt  # положение
    return speed_object, position_object


# Размещение сферы радиуса r в точке x, y, z
def plot_share(ax, x, y, z, r, resolution, **kwargs):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    xx = r * np.outer(np.cos(u), np.sin(v)) + x
    yy = r * np.outer(np.sin(u), np.sin(v)) + y
    zz = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    ax.plot_surface(xx, yy, zz, rstride=4, cstride=4, **kwargs)


# Задаем начальные условия
speed = np.array([1., 0., 0.])  # начальный вектор скорости
position = np.array([0., 1., 0.])  # начальная позиция
sat1 = Satellite(1, speed, position)  # создали объект класса


speed = np.array([1., 0., 0.])  # начальный вектор скорости
position = np.array([0., 1.2, 0.])  # начальная позиция
sat2 = Satellite(1, speed, position)

speed = np.array([0.9, 0., 0.])  # начальный вектор скорости
position = np.array([0., 0.8, 0.])  # начальная позиция
sat3 = Satellite(1, speed, position)

# рисуем
fig = plt.figure('flight_of_satellites')  # Название и размер окна в дюймах
ax = fig.add_subplot(projection='3d')  # Настройка на 3d графику
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

# планета (в центре)
plot_share(ax, 0, 0, 0, 0.2, 50, color=[0.5, 0.5, .5])

# Начальное положение спутника
sat1.plot_sat()
sat2.plot_sat()
sat3.plot_sat()

# Перемещение спутников
def update(_):
    sat1.new_condition()
    sat1.pos.remove()  # Удаление предыдущего положения
    sat1.plot_sat(s=30, c='k')
    sat1.plot_trail()  # Рисуем траекторию

    sat2.new_condition(num_iterations=250)
    sat2.pos.remove()  # Удаление предыдущего положения
    sat2.plot_sat(s=40, c='r')
    sat2.plot_trail()

    sat3.new_condition()
    sat3.pos.remove()  # Удаление предыдущего положения
    sat3.plot_sat(s=50, c='g')
    sat3.plot_trail()

# Запуск анимации
if __name__ == "__main__":
    ani = animation.FuncAnimation(fig, update, frames=200, interval=100)
    """
    * frames=200: Это количество кадров в анимации.
    Функция update будет вызвана 200 раз. Можно передать итератор кадров.
    
    * interval=100: Это время в миллисекундах между кадрами.
        В данном случае, каждый кадр будет отображаться на экране в течение 100 миллисекунд,
    что соответствует частоте 5 кадров в секунду (1000 мс / 100 = 10 кадров/сек).
        Это позволяет снизить нагрузку 
    """
    plt.show()
