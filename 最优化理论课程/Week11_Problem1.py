import numpy as np
import json
import time
import sympy as sym
from sympy import hessian, diff, symbols, solve, lambdify
import matplotlib.pyplot as plt


def time_decorator(func):
    # 装饰器，测量执行时间
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds.")
        return result

    return wrapper


def initial_vector(n) -> np.ndarray:
    return np.tile([1, 2, 1], int(np.ceil((n + 2) / 3)))[:n + 2]


def objective_function(x) -> float:
    n = len(x) - 2
    sum_value = 0
    for i in range(1, n + 1):
        xi = x[i]
        xi1 = x[(i + 1) % n]  # 使用周期边界条件
        xi2 = x[(i + 2) % n]
        sum_value += ((-xi ** 2 + xi1 ** 2 + xi2 ** 2) ** 2 + (xi ** 2 - xi1 ** 2 + xi2 ** 2) ** 2 + (
                xi ** 2 + xi1 ** 2 - xi2 ** 2) ** 2)
    return sum_value


def gradient_sym():
    x1, x2, x3 = sym.symbols('x1 x2 x3')
    fi = (-x1 ** 2 + x2 ** 2 + x3 ** 2) ** 2 + (x1 ** 2 - x2 ** 2 + x3 ** 2) ** 2 + (x1 ** 2 + x2 ** 2 - x3 ** 2) ** 2
    # 可以采用sympy库计算梯度
    # return [diff(fi, x1), diff(fi, x2), diff(fi, x3)]
    # Out[6]: [4*x1*(x1**2 - x2**2 + x3**2) - 4*x1*(x1**2 + x2**2 - x3**2) - 4*x1*(-x1**2 + x2**2 + x3**2),
    #          -4*x2*(-x1**2 + x2**2 + x3**2) + 4*x2*(x1**2 - x2**2 + x3**2) - 4*x2*(x1**2 + x2**2 - x3**2),
    #          4*x3*(x1**2 + x2**2 - x3**2) - 4*x3*(x1**2 - x2**2 + x3**2) - 4*x3*(-x1**2 + x2**2 + x3**2)]
    gradient = sym.Matrix([fi]).jacobian([x1, x2, x3]).T
    return gradient
    #   gradient.subs([(x1,1),(x2,2),(x3,1)]) #在x=1,y=0,z=1处的梯度值


def gradient_calculating(x) -> np.ndarray:
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n - 2):
        xi = x[i]
        xi1 = x[(i + 1) % n]
        xi2 = x[(i + 2) % n]
        # 对于每个xi，计算其影响到的梯度分量
        grad[i] = grad[i] + 12 * xi ** 3 - 4 * xi1 ** 2 * xi - 4 * xi2 ** 2 * xi
        grad[(i + 1) % n] = grad[(i + 1) % n] - 4 * xi ** 2 * xi1 + 12 * xi1 ** 3 - 4 * xi2 ** 2 * xi1
        grad[(i + 2) % n] = grad[(i + 2) % n] - 4 * xi ** 2 * xi2 - 4 * xi1 ** 2 * xi2 + 12 * xi2 ** 3
    return grad.astype(np.float64)


# 用sympy计算符号化的Hessian矩阵
def hessian_sym():
    x1, x2, x3 = sym.symbols('x1 x2 x3')
    fi = (-x1 ** 2 + x2 ** 2 + x3 ** 2) ** 2 + (x1 ** 2 - x2 ** 2 + x3 ** 2) ** 2 + (x1 ** 2 + x2 ** 2 - x3 ** 2) ** 2
    return hessian(fi, (x1, x2, x3))
    # Out[11]:
    # Matrix([
    #     [36 * x1 ** 2 - 4 * x2 ** 2 - 4 * x3 ** 2, -8 * x1 * x2, -8 * x1 * x3],
    #     [-8 * x1 * x2, -4 * x1 ** 2 + 36 * x2 ** 2 - 4 * x3 ** 2, -8 * x2 * x3],
    #     [-8 * x1 * x3, -8 * x2 * x3, -4 * x1 ** 2 - 4 * x2 ** 2 + 36 * x3 ** 2]])
    # hessian_matrix = hessian(fi, (x1, x2, x3))
    # hessian_matrix.subs([(x1,1),(x2,2),(x3,1)])
    # Out[13]:
    # Matrix([
    # [ 16, -16,  -8],
    # [-16, 136, -16],
    # [ -8, -16,  16]])


def hessian_calculating(x) -> np.ndarray:
    n = len(x)
    H = np.zeros([n, n])
    for i in range(n - 2):
        xi = x[i]
        xi1 = x[(i + 1) % n]
        xi2 = x[(i + 2) % n]
        H[i, i] += 36 * xi ** 2 - 4 * xi1 ** 2 - 4 * xi2 ** 2
        H[i, i + 1] += -8 * xi * xi1
        H[i, i + 2] += -8 * xi * xi2
        H[i + 1, i] += -8 * xi * xi1
        H[i + 1, i + 1] += -4 * xi ** 2 + 36 * xi1 ** 2 - 4 * xi2 ** 2
        H[i + 1, i + 2] += -8 * xi1 * xi2
        H[i + 2, i] += -8 * xi * xi2
        H[i + 2, i + 1] += -8 * xi1 * xi2
        H[i + 2, i + 2] += -4 * xi ** 2 - 4 * xi1 ** 2 + 36 * xi2 ** 2
    return H


def is_real_number(x: sym.core.add.Add):
    # 判断x是否为实数或者是否可以近似为实数(x须为sympy的表达式类型，即solve的解)
    x_real, x_img = x.as_real_imag()
    if x_img < 1e-10:
        return x_real
    return complex(x_real, x_img)


@time_decorator
def newtons_method(x0, precision, max_iter) -> tuple:
    x = x0.copy()
    fx = [objective_function(x)]
    for iter in range(max_iter):
        grad_x = gradient_calculating(x)
        if np.linalg.norm(grad_x) <= precision:
            break
        H_x = hessian_calculating(x)
        H_x_inv = np.linalg.inv(H_x)  # 计算Hessian矩阵的逆
        S = -np.dot(H_x_inv, grad_x)  # 计算牛顿方向
        x_new = x + 1 * S
        x = x_new
        if objective_function(x) < max(fx):
            fx.append(objective_function(x))
    return x, fx, objective_function(x), iter


@time_decorator
def amendment_newtons_method(x0, precision, max_iter) -> tuple:
    x = x0.copy()
    fx = [objective_function(x)]
    alpha = symbols('alpha')  # 定义符号变量alpha为搜索步长
    for iter in range(max_iter):
        grad_x = gradient_calculating(x)
        if np.linalg.norm(grad_x) <= precision:
            break
        H_x = hessian_calculating(x)
        H_x_inv = np.linalg.inv(H_x)  # 计算Hessian矩阵的逆
        S = -np.dot(H_x_inv, grad_x)  # 计算牛顿方向
        alpha_val = solve(diff(objective_function(x + S * alpha), alpha), alpha)  # 求解最优步长
        # 解出来的alpha有可能是复数且多个解
        if isinstance(alpha_val, list):
            alpha_val_real = []
            # 遍历解的列表，判断是否为复数，保留实数和可以近似为实数的复数中的实部
            for alpha_val_i in alpha_val:
                # 判断是否为复数
                if alpha_val_i.is_complex:
                    alpha_val_i_real, alpha_val_i_img = alpha_val_i.as_real_imag()  # 分离实部和虚部
                    # 如果虚部过小，则近似为实数，保留其实部
                    if alpha_val_i_img < 1e-10:
                        alpha_val_real.append(alpha_val_i_real)
                else:
                    alpha_val_real.append(alpha_val_i)
            # 计算实数alpha列表中使得partial f(x+alpha*S)/partial alpha的值最接近0的alpha值
            f = lambdify(alpha, diff(objective_function(x + S * alpha), alpha), 'numpy')
            min_alpha = alpha_val_real[0]
            for alpha_val_real_i in alpha_val_real:
                min_alpha = alpha_val_real_i if abs(f(alpha_val_real_i)) < abs(f(min_alpha)) else min_alpha

            alpha_val = min_alpha
        x_new = x + alpha_val * S
        # print(f'{iter}, f(x):{objective_function(x)}, ||g(x)||:{np.linalg.norm(grad_x)}')
        x = x_new
        fx.append(objective_function(x))
    return x, fx, objective_function(x), iter


@time_decorator
def dfp_method(x0, precision, max_iter) -> tuple:
    x = x0.copy()
    fx = [objective_function(x)]
    alpha = symbols('alpha')  # 定义符号变量alpha为搜索步长
    for iter in range(max_iter):
        A = np.eye(len(x))  # 初始化A为单位矩阵
        for k in range(len(x)):
            grad_x = gradient_calculating(x)
            if np.linalg.norm(grad_x) <= precision:
                return x, fx, objective_function(x), iter
            S = -np.dot(A, grad_x)  # 计算牛顿方向
            alpha_val = solve(diff(objective_function(x + S * alpha), alpha), alpha)  # 求解最优步长
            # 解出来的alpha有可能是复数且多个解
            if isinstance(alpha_val, list):
                alpha_val_real = []
                # 遍历解的列表，判断是否为复数，保留实数和可以近似为实数的复数中的实部
                for alpha_val_i in alpha_val:
                    # 判断是否为复数
                    if alpha_val_i.is_complex:
                        alpha_val_i_real, alpha_val_i_img = alpha_val_i.as_real_imag()  # 分离实部和虚部
                        # 如果虚部过小，则近似为实数，保留其实部
                        if alpha_val_i_img < 1e-10:
                            alpha_val_real.append(alpha_val_i_real)
                    else:
                        alpha_val_real.append(alpha_val_i)
                # 计算实数alpha列表中使得partial f(x+alpha*S)/partial alpha的值最接近0的alpha值
                f = lambdify(alpha, diff(objective_function(x + S * alpha), alpha), 'numpy')
                min_alpha = alpha_val_real[0]
                for alpha_val_real_i in alpha_val_real:
                    min_alpha = alpha_val_real_i if abs(f(alpha_val_real_i)) < abs(f(min_alpha)) else min_alpha

                alpha_val = min_alpha
            # 计算\delta x, \delta grad, E
            x_new = x + alpha_val * S
            # print(f'{iter}, f(x):{objective_function(x)}, ||g(x)||:{np.linalg.norm(grad_x)}')
            grad_x_new = gradient_calculating(x_new)
            if np.linalg.norm(grad_x_new - grad_x) < precision:
                break
            grad_x_new = gradient_calculating(x_new)
            delta_grad = grad_x_new - grad_x
            delta_x = x_new - x
            E = np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_grad) - np.dot(np.dot(A, delta_grad),
                                                                                    np.dot(delta_grad.T, A)) / np.dot(
                np.dot(delta_grad.T, A), delta_grad)
            A = A + E
            x = x_new
            fx.append(objective_function(x))
    return x, fx, objective_function(x), iter


@time_decorator
def gradient_bb(x0, precision, max_iter) -> tuple:
    x = x0
    fx = [objective_function(x)]
    s = x  # 初始化s
    y = gradient_calculating(x)
    alpha = 0.01  # 初始化步长
    for i in range(max_iter):
        grad = gradient_calculating(x)
        x_new = x - alpha * grad
        s = x_new - x
        y = gradient_calculating(x_new) - grad
        if np.linalg.norm(grad) < precision:
            break
        # 更新BB步长
        alpha = np.abs(s.dot(s) / s.dot(y))
        x = x_new
        fx.append(objective_function(x))
    return x, fx, objective_function(x), i


def figure_draw(fx_newton, fx_amendment_newton, fx_bb, fx_dfp, n):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(fx_newton, label='Newton Method', marker='.')
    plt.plot(fx_amendment_newton, label='Amendment Newton Method', marker='D')
    plt.plot(fx_bb, label='Gradient BB', marker='x')
    plt.plot(fx_dfp, label='DFP Method', marker='s')
    plt.xlabel('Iterations')
    # x刻度为整数
    plt.ylabel('f(x)')
    plt.title(f'f(x) in each iterations when n={n}')
    plt.legend()
    plt.savefig(f'figures/Week11_Problem1_{n}.png')
    plt.show()


def main():
    precision = 0.001
    max_iter = 1000
    # 测试并比较算法性能
    # n_values = [100]
    n_values = [1, 10, 100]
    results = {}
    for n in n_values:
        print('---------- n =', n, '----------')
        x0 = initial_vector(n)

        x_newton, fx_newton, fx_newton_best, iter_newton = newtons_method(x0, precision=precision, max_iter=max_iter)
        x_amendment_newton, fx_amendment_newton, fx_amendment_newton_best, iter_amendment_newton \
            = amendment_newtons_method(x0, precision=precision, max_iter=max_iter)
        x_bb, fx_bb, fx_bb_best, iter_bb = gradient_bb(x0, precision=precision, max_iter=max_iter)
        x_dfp, fx_dfp, fx_dfp_best, iter_dfp = dfp_method(x0, precision=precision, max_iter=max_iter)

        figure_draw(fx_newton, fx_amendment_newton, fx_bb, fx_dfp, n)

        results[n] = {
            'Newton Method': {
                'min f(x)': float(fx_newton_best),
                'iterations': int(iter_newton),
            },
            'Amendment Newton Method': {
                'min f(x)': float(fx_amendment_newton_best),
                'iterations': int(iter_amendment_newton),
            },
            'Gradient BB': {
                'min f(x)': float(fx_bb_best),
                'iterations': int(iter_bb),
            },
            'DFP Method': {
                'min f(x)': float(fx_dfp_best),
                'iterations': int(iter_dfp),
            }
        }

    # 格式化并打印结果
    print('************* Results *************')
    print(json.dumps(results, indent=4, ensure_ascii=False))
    return results


if __name__ == '__main__':
    print('************* Student Info:周聪杰 2023320397 *************')
    main()
