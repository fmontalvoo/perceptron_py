from perceptron import Perceptron

if __name__ == "__main__":
    entradas = [[0.73, 0.21], [1.21, 1.7], [2.07, 3.7], [2.33, 2.7]]
    delta = [1, 1, 0, 0]
    pesos = [-0.23, -1.87]
    bias = 1.31
    tasa_aprendizaje = 0.91

    # entradas = [[1.53, 0.53], [1.5, 1], [2, 1.73], [3, 1.5]]
    # delta = [1, 1, 0, 0]
    # pesos = [0.23, -0.1]
    # bias = 1.47
    # tasa_aprendizaje = 1

    perceptron = Perceptron(entradas, delta, pesos, bias, tasa_aprendizaje)
    perceptron.entrenar()
    print(f"\n{'*'*30}[Resultado]{'*'*30}")
    print(f'\nW: {perceptron.pesos} | Bias: {perceptron.bias}\n')
    x, f_x = perceptron.recta(perceptron.x_y())
    perceptron.graficar(x, f_x)

    print(perceptron.predecir([1.75, 3]))
    print(perceptron.predecir([1.5, 1.5]))