import numpy as np
import json
import time
from sko.GA import GA


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
    n = len(x) - 2
    sum_value = 0
    for i in range(1, n + 1):
        xi = x[i]
        xi1 = x[(i + 1) % n]  # 使用周期边界条件
        xi2 = x[(i + 2) % n]
        sum_value += ((-xi + xi1 + xi2) ** 2 + (xi - xi1 + xi2) ** 2 + (xi + xi1 - xi2) ** 2)
    return sum_value


@time_decorator
def classic_SPSA_algorithm(x0, max_iter, a, c) -> np.ndarray:
    x = x0.copy()
    n = len(x) - 2
    for i in range(max_iter):
        delta = 2 * np.random.randint(0, 2, size=n + 2) - 1  # 生成随机扰动
        x_plus = x + c * delta
        x_minus = x - c * delta
        f_plus = objective_function(x_plus)
        f_minus = objective_function(x_minus)
        gradient_estimate = (f_plus - f_minus) / (2 * c * delta)  # 计算梯度估计
        step_size = a / (i + 1)
        x = x - step_size * gradient_estimate  # 更新x
    return x


@time_decorator
def SPSA_algorithm_v2(x0, n_iter, a, c, A)->np.ndarray:
    x = x0.copy()
    n = len(x) - 2
    for i in range(n_iter):
        delta = 2 * np.random.randint(0, 2, size=n + 2) - 1
        x_plus = x + A * c * delta
        x_minus = x - A * c * delta
        f_plus = objective_function(x_plus)
        f_minus = objective_function(x_minus)
        gradient_estimate = (f_plus - f_minus) / (2 * A * c * delta)
        step_size = a / (i + 1)
        x = x - step_size * gradient_estimate
        # 更新A的值
        if i % 10 == 0 and i > 0:  # 每隔一定迭代次数更新一次A的值
            if abs(f_plus - f_minus) < 0.01*(f_minus+f_plus)/2:  # 如果目标函数值变化较小，则减小A
                A *= 0.9
            else:  # 如果目标函数值变化较大，则增加A
                A *= 1.1
    return x


@time_decorator
def ga(n, max_iter)->tuple:
    my_ga = GA(func=objective_function, n_dim=n, size_pop=50, max_iter=max_iter, precision=1e-7)
    x_ga, y_ga = my_ga.run()
    return x_ga, y_ga


def main():
    # 设置SPSA参数
    n_values = [100, 1000, 10000]
    max_iter = 1000
    a = 0.02
    c = 0.005
    A = 0.9


    results = {}

    # 测试SPSA算法
    for n in n_values:
        print('---------- n =', n, '----------')
        x0 = np.tile([1, 2, 3], n + 2)
        x_classic = classic_SPSA_algorithm(x0, max_iter, a, c)
        x_v3 = SPSA_algorithm_v2(x0, max_iter, a, c, A)
        _, y_ga = ga(n, max_iter)
        results[n] = {
            'classic_SPSA': objective_function(x_classic),
            'SPSA_v3': objective_function(x_v3),
            'GA': y_ga.tolist(),
        }

    # 格式化并打印结果
    print('************* Results *************')
    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    print('************* Student Info:周聪杰 2023320397 *************')
    main()
