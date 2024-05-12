import numpy as np
import json
import time
from sympy import symbols, diff, solve



def time_decorator(func):
    # 装饰器，测量执行时间
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds.")
        return result

    return wrapper


def objective_function(x) -> float:
    n = len(x)
    f_val = 0
    for i in range(n):
        xi = x[i]
        xi1 = x[(i + 1) % n]  # 使用周期边界条件
        xi2 = x[(i + 2) % n]
        f_val += ((-xi + xi1 + xi2) ** 2 + (xi - xi1 + xi2) ** 2 + (xi + xi1 - xi2) ** 2)
    return f_val


def gradient(x) -> np.ndarray:
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        xi = x[i]
        xi1 = x[(i + 1) % n]
        xi2 = x[(i + 2) % n]
        # 对于每个xi，计算其影响到的梯度分量
        grad[i] = 2 * ((-xi + xi1 + xi2) * (-1) +
                       (xi - xi1 + xi2) * 1 +
                       (xi + xi1 - xi2) * 1)
        grad[(i + 1) % n] += 2 * ((-xi + xi1 + xi2) * 1 +
                                  (xi - xi1 + xi2) * (-1) +
                                  (xi + xi1 - xi2) * 1)
        grad[(i + 2) % n] += 2 * ((-xi + xi1 + xi2) * 1 +
                                  (xi - xi1 + xi2) * 1 +
                                  (xi + xi1 - xi2) * (-1))
    return grad


# 初始化向量
def initial_vector(n) -> np.ndarray:
    return np.tile([1, 2, 3], int(np.ceil((n + 2) / 3)))[:n + 2]


@time_decorator
# 梯度下降法
def gradient_descent(x0, precision, max_iter) -> tuple:
    x = x0
    alpha = symbols('alpha')  # 定义符号变量alpha为搜索步长
    for i in range(max_iter):
        grad = - gradient(x)
        alpha_val = solve(diff(objective_function(x + grad * alpha), alpha), alpha)  # 求解最优步长
        # 求解最优步长是关键步骤.实测使用固定步长(alpha=0.001)与每次计算最优步长，
        # n=100时，前者计算时间短(0.0832s)但迭代519次收敛，后者计算时间长达(37.7206s)但迭代次数仅为52次
        alpha_val = float(alpha_val[0])  # 将解从分数转换为浮点数
        x_new = x + alpha_val * grad
        if np.linalg.norm(grad) < precision:  # 判断梯度范数是否达到精度要求
            break
        x = x_new
    return x, i


# 共轭梯度法
@time_decorator
def conjugate_gradient(x0, precision, max_iter) -> tuple:
    x = x0
    alpha = symbols('alpha')  # 定义符号变量alpha为搜索步长
    for i in range(max_iter):
        grad = []
        for k in range(len(x)):
            grad = - gradient(x) if len(grad)==0 else grad
            alpha_val = solve(diff(objective_function(x + grad * alpha), alpha), alpha)  # 求解最优步长
            alpha_val = float(alpha_val[0])  # 将解从分数转换为浮点数
            x_new = x + alpha_val * grad
            grad_new = gradient(x_new)
            x = x_new
            if np.linalg.norm(grad_new) < precision:  # 判断梯度范数是否达到精度要求
                return x, i
            else:
                beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
                grad = -grad_new + beta * grad

    return x, i


@time_decorator
# BB方法
def gradient_bb(x0, precision, max_iter) -> tuple:
    x = x0
    s = x  # 初始化s
    y = gradient(x)
    alpha = 0.01  # 初始化步长
    for i in range(max_iter):
        grad = gradient(x)
        x_new = x - alpha * grad
        s = x_new - x
        y = gradient(x_new) - grad
        if np.linalg.norm(grad) < precision:
            break
        # 更新BB步长
        alpha = np.abs(s.dot(s) / s.dot(y))
        x = x_new
    return x, i


def main() -> None:
    precision = 0.001
    max_iter = 1000
    # 测试并比较算法性能
    n_values = [1, 10, 100]
    results = {}
    for n in n_values:
        print('---------- n =', n, '----------')
        x0 = initial_vector(n)
        result_gd, iter_gd = gradient_descent(x0, precision=precision, max_iter=max_iter)
        result_cg,iter_cg = conjugate_gradient(x0, precision=precision, max_iter=max_iter)
        result_bb, iter_bb = gradient_bb(x0, precision=precision, max_iter=max_iter)
        results[n] = {
            'Gradient Descent': {'result': result_gd.tolist(),
                                 'iterations': iter_gd,
                                 'min f(x)': objective_function(result_gd)},
            'Conjugate Gradient': {'result': result_cg.tolist(),
                                   'iterations': iter_cg,
                                   'min f(x)': objective_function(result_cg)},
            'Gradient BB': {'result': result_bb.tolist(),
                            'iterations': iter_bb,
                            'min f(x)': objective_function(result_bb)},
        }

    # 格式化并打印结果
    print('************* Results *************')
    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    print('************* Student Info:周聪杰 2023320397 *************')
    main()
