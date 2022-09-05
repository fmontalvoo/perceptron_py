import math
import matplotlib.pyplot as pp


class Perceptron:
    _bias: float
    _pesos: list
    _entradas: list
    _salida_esperada: float
    _tasa_aprendizaje: float

    def __init__(self, entradas, salida_esperada, pesos, bias, tasa_aprendizaje):
        self._entradas = entradas
        self._salida_esperada = salida_esperada
        self._pesos = pesos
        self._bias = bias
        self._tasa_aprendizaje = tasa_aprendizaje
        self.entrenar(self._entradas, self._salida_esperada,
                      self._pesos, self._bias, self._tasa_aprendizaje)

    def entrenar(self, entradas, salida_esperada, pesos, bias, tasa_aprendizaje):
        idx = 0
        cont = 0
        continua = True
        while continua:
            if idx >= len(entradas):
                continua = False
            else:
                n = self.n(pesos, entradas[idx], bias)
                f_n = self.f_n(n)
                error = self.error(salida_esperada[idx], f_n)
                print(
                    f'X: {entradas[idx]} | W: {pesos} | f({n})={f_n} | salida_esperada: {salida_esperada[idx]} | Error: {error}')
                if error != 0:
                    pesos = self.ajusta_weight(
                        pesos,
                        tasa_aprendizaje,
                        error,
                        entradas[idx]
                    )
                    bias = self.ajusta_bias(bias, tasa_aprendizaje, error)
                    print(
                        f'W( t+{cont + 1} ): {pesos} | Bias( t+{cont + 1} ): {bias}')
                    idx = 0
                    cont += 1
                else:
                    idx += 1

        print(f"\n{'*'*30}[Resultado]{'*'*30}")
        print(f'\nW: {pesos} | Bias: {bias}\n')
        x, f_x = self.recta(self.x_y(pesos, bias))
        self.graficar(entradas, salida_esperada, x, f_x)

    def n(self, pesos, entradas, bias):
        n_i = []
        for i in range(len(pesos)):
            n_i.append(float(pesos[i] * entradas[i]))
        return float(f'{(sum(n_i) + bias):.4f}')

    def f_n(self, n):
        if n > 0:
            return 1
        else:
            return 0

    def ajusta_weight(self, pesos, tasa_aprendizaje, error, entradas):
        w_i = []
        for i in range(len(pesos)):
            w = pesos[i] + tasa_aprendizaje * error * entradas[i]
            w_i.append(float(f'{w:.4f}'))
        return w_i

    def ajusta_bias(self, bias, tasa_aprendizaje, error):
        return float(f'{(bias + tasa_aprendizaje * error):.4f}')

    def error(self, salida_esperada, f_n):
        return (salida_esperada - f_n)

    def x_y(self, pesos, bias):
        xy = []
        for i in range(len(pesos)):
            if (i % 2 == 0):
                xy.append((float(f'{(-bias / pesos[i]):.4f}'), 0))
            else:
                xy.append((0, float(f'{(-bias / pesos[i]):.4f}')))
        return xy

    def recta(self, xy):
        x, y = xy
        x_min = math.floor(min(self._entradas)[0])
        x_max = math.ceil(1.5 * max(self._entradas)[0])

        m = (y[1] - y[0]) / (x[1] - x[0])
        x_ = []
        f_x = []
        for i in range(x_min, x_max):
            x_.append(i)
            f_x.append(m * i - x[0] * m)
        return (x_, f_x)

    def graficar(self, entradas, salida_esperada, x, f_x):
        x_in = []
        y_in = []

        for coords in entradas:
            a, b = coords
            x_in.append(a)
            y_in.append(b)

        pp.scatter(
            x_in,
            y_in,
            color=['blue' if i == 1 else 'red' for i in salida_esperada]
        )
        pp.plot(x, f_x, color='black')
        pp.title('Salida del perceptrón ')
        pp.grid()
        pp.show()
