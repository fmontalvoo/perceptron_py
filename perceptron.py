import math
import matplotlib.pyplot as pp


class Perceptron:

    def __init__(self, entradas: list[float], salida_esperada: float, pesos: list[float], bias: float, tasa_aprendizaje: float):
        self.entradas = entradas
        self.salida_esperada = salida_esperada
        self.pesos = pesos
        self.bias = bias
        self.tasa_aprendizaje = tasa_aprendizaje

    def entrenar(self):
        idx = 0
        cont = 0
        continua = True
        while continua:
            if idx >= len(self.entradas):
                continua = False
            else:
                n = self.sumatoria(self.entradas[idx])
                f_n = self.funcion_activacion(n)
                error = self.calcular_error(self.salida_esperada[idx], f_n)
                print(
                    f'X: {self.entradas[idx]} | W: {self.pesos} | f({n})={f_n} | Delta: {self.salida_esperada[idx]} | Error: {error}')
                if error != 0:
                    self.pesos = self.calcular_pesos(self.entradas[idx], error)
                    self.bias = self.calcular_bias(error)
                    print(
                        f'\nW( t+{cont + 1} ): {self.pesos} | Bias( t+{cont + 1} ): {self.bias}\n')
                    idx = 0
                    cont += 1
                else:
                    idx += 1

    def predecir(self, entradas: list[float]) -> int:
        n = self.sumatoria(entradas)
        f_n = self.funcion_activacion(n)
        return f_n

    def sumatoria(self, entradas: list[float]) -> float:
        suma = self.bias
        for i in range(len(self.pesos)):
            suma += (self.pesos[i] * entradas[i])
        return float(f'{suma:.4f}')

    def funcion_activacion(self, n: float) -> int:
        return 1 if n > 0 else 0

    def calcular_pesos(self, entradas: list[float], error: int) -> list[float]:
        w_i = []
        for i in range(len(self.pesos)):
            w = self.pesos[i] + self.tasa_aprendizaje * error * entradas[i]
            w_i.append(float(f'{w:.4f}'))
        return w_i

    def calcular_bias(self, error: int) -> float:
        return float(f'{(self.bias + self.tasa_aprendizaje * error):.4f}')

    def calcular_error(self, salida_esperada: list[float], f_n: int) -> int:
        return salida_esperada - f_n

    def calcular_recta(self) -> tuple[list[float], list[float]]:
        x = (float(f'{(-self.bias / self.pesos[0]):.4f}'), 0)
        y = (0, float(f'{(-self.bias / self.pesos[1]):.4f}'))

        x_min = math.floor(min(self.entradas)[0])
        x_max = math.ceil(1.5 * max(self.entradas)[0])

        m: float = (y[1] - y[0]) / (x[1] - x[0])
        x_: list[float] = []
        f_x: list[float] = []
        for i in range(x_min, x_max):
            x_.append(i)
            f_x.append(m * i - x[0] * m)
        return (x_, f_x)

    def graficar(self, x: list[float], f_x: list[float]):
        x_in: list[float] = []
        y_in: list[float] = []

        for coords in self.entradas:
            a, b = coords
            x_in.append(a)
            y_in.append(b)

        pp.scatter(
            x_in,
            y_in,
            color=['blue' if i == 1 else 'red' for i in self.salida_esperada]
        )
        pp.plot(x, f_x, color='black')
        pp.title('Salida del perceptrón ')
        pp.grid()
        pp.show()
