import numpy as np
import json
import time
import sympy as sym
from sympy import hessian, diff, symbols, solve, lambdify
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


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


def total_objective_function(x) -> float:
    n = len(x) - 2
    sum_value = 0
    for i in range(1, n + 1):
        xi = x[i]
        xi1 = x[(i + 1) % n]  # 使用周期边界条件
        xi2 = x[(i + 2) % n]
        sum_value += ((-xi ** 2 + xi1 ** 2 + xi2 ** 2) ** 2 + (xi ** 2 - xi1 ** 2 + xi2 ** 2) ** 2 + (
                xi ** 2 + xi1 ** 2 - xi2 ** 2) ** 2)
    return sum_value


def block_objective_function(x_all, k) -> float:
    x_block = x_all[k * 3:k * 3 + 3]
    if k > 0:
        if k < len(x_all) / 3:
            x_related = x_all[k * 3 - 2: k * 3 + 3 + 2]
        else:
            x_related = x_all[k * 3 - 2:]
    else:
        x_related = x_block
    x = x_related.copy()
    n = len(x_related)
    sum_value = 0
    for i in range(0, n - 2):
        xi = x[i]
        xi1 = x[i + 1]  # 使用周期边界条件
        xi2 = x[i + 2]
        sum_value += ((-xi ** 2 + xi1 ** 2 + xi2 ** 2) ** 2 + (xi ** 2 - xi1 ** 2 + xi2 ** 2) ** 2 + (
                xi ** 2 + xi1 ** 2 - xi2 ** 2) ** 2)
    return sum_value


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


def block_gradient_calculating(x_all, k) -> np.ndarray:
    # x2, x3,x4,x5,x6,x7,x8 = sym.symbols('x2:9')
    # fi =
    # (-x2 ** 2 + x3 ** 2 + x4 ** 2) ** 2 + (x2 ** 2 - x3 ** 2 + x4 ** 2) ** 2 + (x2 ** 2 + x3 ** 2 - x4 ** 2) ** 2 +
    # (-x3 ** 2 + x4 ** 2 + x5 ** 2) ** 2 + (x3 ** 2 - x4 ** 2 + x5 ** 2) ** 2 + (x3 ** 2 + x4 ** 2 - x5 ** 2) ** 2 +
    # (-x4 ** 2 + x5 ** 2 + x6 ** 2) ** 2 + (x4 ** 2 - x5 ** 2 + x6 ** 2) ** 2 + (x4 ** 2 + x5 ** 2 - x6 ** 2) ** 2 +
    # (-x5 ** 2 + x6 ** 2 + x7 ** 2) ** 2 + (x5 ** 2 - x6 ** 2 + x7 ** 2) ** 2 + (x5 ** 2 + x6 ** 2 - x7 ** 2) ** 2 +
    # (-x6 ** 2 + x7 ** 2 + x8 ** 2) ** 2 + (x6 ** 2 - x7 ** 2 + x8 ** 2) ** 2 + (x6 ** 2 + x7 ** 2 - x8 ** 2) ** 2
    # gradient = sym.Matrix([fi]).jacobian([x4, x5, x6]).T
    # simplify(gradient)
    # Out[8]:
    # Matrix([
    #     [4 * x4 * (-x2 ** 2 - 2 * x3 ** 2 + 9 * x4 ** 2 - 2 * x5 ** 2 - x6 ** 2)],
    #     [4 * x5 * (-x3 ** 2 - 2 * x4 ** 2 + 9 * x5 ** 2 - 2 * x6 ** 2 - x7 ** 2)],
    #     [4 * x6 * (-x4 ** 2 - 2 * x5 ** 2 + 9 * x6 ** 2 - 2 * x7 ** 2 - x8 ** 2)]])
    x_block = x_all[k * 3:k * 3 + 3]
    if k > 0:
        if k < len(x_all) / 3 - 1:
            x_related = x_all[k * 3 - 2: k * 3 + 3 + 2]
            x = x_related.copy()
            grad = np.zeros_like(x_block)
            grad[0] = 4 * x[0] * (-x[0] ** 2 - 2 * x[1] ** 2 + 9 * x[2] ** 2 - 2 * x[3] ** 2 - x[4] ** 2)
            grad[1] = 4 * x[1] * (-x[1] ** 2 - 2 * x[2] ** 2 + 9 * x[3] ** 2 - 2 * x[4] ** 2 - x[5] ** 2)
            grad[2] = 4 * x[2] * (-x[2] ** 2 - 2 * x[3] ** 2 + 9 * x[4] ** 2 - 2 * x[5] ** 2 - x[6] ** 2)
        else:
            # Out[21]:
            # Matrix([
            #     [4 * x3 * (-x1 ** 2 - 2 * x2 ** 2 + 9 * x3 ** 2 - 2 * x4 ** 2 - x5 ** 2)],
            #     [4 * x4 * (-x2 ** 2 - 2 * x3 ** 2 + 6 * x4 ** 2 - x5 ** 2)],
            #     [4 * x5 * (-x3 ** 2 - x4 ** 2 + 3 * x5 ** 2)]])
            x_related = x_all[k * 3 - 2:]
            x = x_related.copy()
            grad = np.zeros_like(x_block)
            grad[0] = 4 * x[2] * (-x[0] ** 2 - 2 * x[1] ** 2 + 9 * x[2] ** 2 - 2 * x[3] ** 2 - x[4] ** 2)
            grad[1] = 4 * x[3] * (-x[1] ** 2 - 2 * x[2] ** 2 + 6 * x[3] ** 2 - x[4] ** 2)
            grad[2] = 4 * x[4] * (-x[2] ** 2 - x[3] ** 2 + 3 * x[4] ** 2)
    else:
        x_related = x_all[0:5]
        x = x_related.copy()
        grad = np.zeros_like(x_block)
        # Out[13]:
        # Matrix([
        #     [4 * x2 * (3 * x2 ** 2 - x3 ** 2 - x4 ** 2)],
        #     [4 * x3 * (-x2 ** 2 + 6 * x3 ** 2 - 2 * x4 ** 2 - x5 ** 2)],
        #     [4 * x4 * (-x2 ** 2 - 2 * x3 ** 2 + 9 * x4 ** 2 - 2 * x5 ** 2 - x6 ** 2)]])
        grad[0] = 4 * x[0] * (3 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2)
        grad[1] = 4 * x[1] * (-x[0] ** 2 + 6 * x[1] ** 2 - 2 * x[2] ** 2 - x[3] ** 2)
        grad[2] = 4 * x[2] * (-x[0] ** 2 - 2 * x[1] ** 2 + 9 * x[2] ** 2 - 2 * x[3] ** 2 - x[4] ** 2)
    return grad.astype(np.float64)


# 以下为BCD算法的更新格式1
def block_function_type_1(x_all, k) -> float:
    return block_objective_function(x_all, k)


def gradient_descent_type_1(x_all, k, precision, max_iter) -> tuple:
    for i in range(max_iter):
        grad = np.zeros_like(x_all)
        grad_k = - block_gradient_calculating(x_all, k)
        grad[k * 3:k * 3 + 3] = grad_k

        # block_function = block_function_type_1(x_all + grad * alpha, k)
        # alpha_val = solve(diff(block_function, alpha), alpha)  # 求解最优步长
        # alpha_val_real = get_real_alpha(alpha_val)
        # alpha_val = get_best_alpha(alpha_val_real, grad, block_function)

        objective_func_alpha = lambda alpha: block_function_type_1(x_all + grad * alpha, k)
        res = minimize_scalar(objective_func_alpha)
        alpha_val = res.x

        x_all_new = x_all + alpha_val * grad
        grad_k_new = - block_gradient_calculating(x_all_new, k)
        if np.linalg.norm(grad_k) < precision or np.linalg.norm(grad_k - grad_k_new) < 0.01:  # 判断梯度范数是否达到精度要求
            x_all = x_all_new
            break
        x_all = x_all_new
    return x_all


@time_decorator
def BCD_type_1(x, precision, max_iter):
    x_all = x.copy()
    datas = {}
    fx = [total_objective_function(x_all)]
    for iteration in range(max_iter):
        datas[f'iteration{iteration}'] = {}
        for k in range(int(len(x_all) / 3)):
            grad = gradient_calculating(x_all)
            x_all_new = gradient_descent_type_1(x_all, k, precision, max_iter)
            grad_new = gradient_calculating(x_all_new)

            datas[f'iteration{iteration}'][f'k{k}'] = {'f(x)': total_objective_function(x_all_new),
                                                       '||g(x)||': np.linalg.norm(grad_new)}
            if np.linalg.norm(grad_new) < precision or np.linalg.norm(grad - grad_new) < precision:
                return total_objective_function(x_all), iteration, datas, fx
            x_all = x_all_new
        fx.append(total_objective_function(x_all))
    return total_objective_function(x_all), iteration, datas, fx


# 以下为BCD算法的更新格式2
def block_function_type_2(x_all, k, x_all_old) -> float:
    L = 1
    x_k = x_all[k * 3:k * 3 + 3]
    x_k_1 = x_all_old[k * 3:k * 3 + 3] if k > 0 else x_k
    mid_term = L / 2 * np.linalg.norm(x_k - x_k_1)

    return block_function_type_1(x_all, k) + mid_term


def gradient_descent_type_2(x_all, k, precision, max_iter) -> tuple:
    x_old = x_all.copy()
    for i in range(max_iter):
        grad = np.zeros_like(x_all)
        grad_k = - block_gradient_calculating(x_all, k)
        grad[k * 3:k * 3 + 3] = grad_k

        objective_func_alpha = lambda alpha: block_function_type_2(x_all + grad * alpha, k, x_old)
        res = minimize_scalar(objective_func_alpha)
        alpha_val = res.x

        x_all_new = x_all + alpha_val * grad
        grad_k_new = - block_gradient_calculating(x_all_new, k)
        if np.linalg.norm(grad_k) < precision or np.linalg.norm(grad_k - grad_k_new) < 0.01:  # 判断梯度范数是否达到精度要求
            x_all = x_all_new
            break
        x_old = x_all.copy()
        x_all = x_all_new
    return x_all


@time_decorator
def BCD_type_2(x, precision, max_iter):
    x_all = x.copy()
    datas = {}
    fx = [total_objective_function(x_all)]
    for iteration in range(max_iter):
        datas[f'iteration{iteration}'] = {}
        for k in range(int(len(x_all) / 3)):
            grad = gradient_calculating(x_all)
            x_all_new = gradient_descent_type_2(x_all, k, precision, max_iter)
            grad_new = gradient_calculating(x_all_new)
            # print(
            #     f'iter: {iteration}, k: {k},f(x):{total_objective_function(x_all_new)},  ||g(x)||: {np.linalg.norm(grad_new)}')
            datas[f'iteration{iteration}'][f'k{k}'] = {'f(x)': total_objective_function(x_all_new),
                                                       '||g(x)||': np.linalg.norm(grad_new)}
            if np.linalg.norm(grad_new) < precision or np.linalg.norm(grad - grad_new) < precision:
                return total_objective_function(x_all), iteration, datas, fx
            x_all = x_all_new
        fx.append(total_objective_function(x_all))
    return total_objective_function(x_all), iteration, datas, fx


# 以下为BCD算法的更新格式3
def block_function_type_3(x_all, k, x_all_old, x_old_2, grad_old) -> float:
    omega = 0
    x_k = x_all[k * 3:k * 3 + 3]

    x_k_old = x_all_old[k * 3:k * 3 + 3]
    x_k_old_2 = x_old_2[k * 3:k * 3 + 3]

    hat_x_k_old = x_k_old + omega * (x_k_old - x_k_old_2)

    grad_last = grad_old[k * 3:k * 3 + 3]
    first_term = np.dot(grad_last, x_k - hat_x_k_old)

    mid_term = 0.5 * np.linalg.norm(x_k - hat_x_k_old) ** 2

    fi_ri = block_objective_function(x_all, k)
    fi = ((-x_k[0] ** 2 + x_k[1] ** 2 + x_k[2] ** 2) ** 2 + (x_k[0] ** 2 - x_k[1] ** 2 + x_k[2] ** 2) ** 2 + (
            x_k[0] ** 2 + x_k[1] ** 2 - x_k[2] ** 2) ** 2)
    ri = fi_ri - fi

    return first_term + mid_term + ri


def gradient_descent_type_3(x_all, k, precision, max_iter, x_old, x_old_2, grad_last) -> tuple:
    for i in range(max_iter):
        grad = np.zeros_like(x_all)
        grad_k = - block_gradient_calculating(x_all, k)
        grad_k_2 = np.linalg.norm(grad_k)
        grad[k * 3:k * 3 + 3] = grad_k

        if i > 1:
            objective_func_alpha = lambda alpha: block_function_type_3(x_all + grad * alpha, k, x_old, x_old_2,
                                                                       grad_last)
        else:
            objective_func_alpha = lambda alpha: block_function_type_1(x_all + grad * alpha, k)

        res = minimize_scalar(objective_func_alpha)
        if res.success:
            alpha_val = res.x
            x_all_new = x_all + alpha_val * grad
        else:
            break

        grad_k_new = - block_gradient_calculating(x_all_new, k)
        if np.linalg.norm(grad_k) < precision or np.linalg.norm(grad_k - grad_k_new) < 0.01:  # 判断梯度范数是否达到精度要求
            x_all = x_all_new
            break
        grad_old = -block_gradient_calculating(x_all, k)
        x_all = x_all_new
    return x_all


def find_new_x_type_3_gradient_BB(x_all, x_old, x_old_2, k):
    omega = 1
    x_all_new = x_all.copy()
    # x_k = x_all[k * 3:k * 3 + 3]

    x_k_old = x_old[k * 3:k * 3 + 3]
    x_k_old_2 = x_old_2[k * 3:k * 3 + 3]
    hat_x_k_old = x_k_old + omega * (x_k_old - x_k_old_2)

    hat_x_all = x_all.copy()
    hat_x_all[k * 3:k * 3 + 3] = hat_x_k_old
    g_i_k = block_gradient_calculating(hat_x_all, k)

    def find_x(x_k: np.ndarray):
        first_term = np.dot(g_i_k, x_k - hat_x_k_old)
        mid_term = 0.5 * np.linalg.norm(x_k - hat_x_k_old) ** 2
        fi_ri = block_objective_function(x_all, k)
        fi = ((-x_k[0] ** 2 + x_k[1] ** 2 + x_k[2] ** 2) ** 2 + (x_k[0] ** 2 - x_k[1] ** 2 + x_k[2] ** 2) ** 2 + (
                x_k[0] ** 2 + x_k[1] ** 2 - x_k[2] ** 2) ** 2)
        ri = fi_ri - fi
        return first_term + mid_term + ri

    def approx_gradient(f, x, eps=1e-6):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.array(x, dtype=float)
            x2 = np.array(x, dtype=float)
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (f(x1) - f(x2)) / (2 * eps)
        return grad

    def gradient_bb(x_all, k, precision, max_iter) -> tuple:
        x_all_new = x_all.copy()
        x_k = x_all[k * 3:k * 3 + 3]
        s = x_k  # 初始化s
        y = approx_gradient(find_x, x_k)
        alpha = 0.01  # 初始化步长
        for i in range(max_iter):
            grad = approx_gradient(find_x, x_k)

            x_k_new = x_k - alpha * grad
            x_all_new[k * 3:k * 3 + 3] = x_k_new
            s = x_k_new - x_k
            y = approx_gradient(find_x, x_k_new) - grad
            if np.linalg.norm(grad) < precision:
                break
            # 更新BB步长
            alpha = np.abs(s.dot(s) / s.dot(y))
            x_k = x_k_new
            x_all = x_all_new
        return x_all

    x_all_new = gradient_bb(x_all, k, 1e-6, 100)
    return x_all_new


# 使用估计梯度下降法以格式3的方式更新x
def find_new_x_type_3_gradient_descent(x_all, x_old, x_old_2, k):
    omega = 1
    x_all_new = x_all.copy()
    # x_k = x_all[k * 3:k * 3 + 3]

    x_k_old = x_old[k * 3:k * 3 + 3]
    x_k_old_2 = x_old_2[k * 3:k * 3 + 3]
    hat_x_k_old = x_k_old + omega * (x_k_old - x_k_old_2)

    hat_x_all = x_all.copy()
    hat_x_all[k * 3:k * 3 + 3] = hat_x_k_old
    g_i_k = block_gradient_calculating(hat_x_all, k)

    def find_x(x_k_1, x_k_2, x_k_3):
        x_k = [x_k_1, x_k_2, x_k_3]
        first_term = np.dot(g_i_k, x_k - hat_x_k_old)
        mid_term = 0.5 * np.linalg.norm(x_k - hat_x_k_old) ** 2
        fi_ri = block_objective_function(x_all, k)
        fi = ((-x_k[0] ** 2 + x_k[1] ** 2 + x_k[2] ** 2) ** 2 + (x_k[0] ** 2 - x_k[1] ** 2 + x_k[2] ** 2) ** 2 + (
                x_k[0] ** 2 + x_k[1] ** 2 - x_k[2] ** 2) ** 2)
        ri = fi_ri - fi
        return first_term + mid_term + ri

    def approx_gradient(f, x, eps=1e-6):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.array(x, dtype=float)
            x2 = np.array(x, dtype=float)
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (f(x1) - f(x2)) / (2 * eps)
        return grad

    def gradient_descent(f, x0, alpha=0.01, max_iter=100):
        x_k = x0
        for i in range(max_iter):
            grad = approx_gradient(f, x_k)  # 确保 f 函数正确处理参数
            x_k = x_k - alpha * grad
            if np.linalg.norm(grad) < 1e-6:
                break
        return x_k

    def function_to_minimize(x_k):
        return find_x(x_k[0], x_k[1], x_k[2])

    # 初始猜测
    x0 = x_all[k * 3:k * 3 + 3]

    # 运行梯度下降
    result_x_k = gradient_descent(function_to_minimize, x0)
    x_all_new[k * 3:k * 3 + 3] = result_x_k

    return x_all_new


@time_decorator
def BCD_type_3(x, precision, max_iter):
    datas = {}
    x_all = x.copy()
    x_old = x_all.copy()
    x_old_2 = x_all.copy()
    fx = [total_objective_function(x_all)]
    for iteration in range(max_iter):
        grad = gradient_calculating(x_all)
        grad_last = grad
        datas[f'iteration{iteration}'] = {}
        for k in range(int(len(x_all) / 3)):
            grad = gradient_calculating(x_all)
            # x_all_new=find_new_x_type_3(x_all, x_old, x_old_2, k)
            x_all_new = gradient_descent_type_3(x_all, k, precision, max_iter, x_old, x_old_2, grad_last)
            grad_new = gradient_calculating(x_all_new)
            datas[f'iteration{iteration}'][f'k{k}'] = {'f(x)': total_objective_function(x_all_new),
                                                       '||g(x)||': np.linalg.norm(grad_new)}
            if np.linalg.norm(grad_new) < precision:
                return total_objective_function(x_all), iteration, datas, fx
            grad_last = grad_new
            x_old_2 = x_old.copy()
            x_old = x_all.copy()
            x_all = x_all_new
        fx.append(total_objective_function(x_all))
    return total_objective_function(x_all), iteration, datas, fx


def figure_draw_datas(n, datas, type):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'

    for iteration, k_values in datas.items():
        ks = []  # 存储k值（迭代步骤）
        fxs = []  # 存储每个k对应的f(x)值
        # 按照k值的数值顺序排序，因为字典的键是字符串形式
        sorted_k_items = sorted(k_values.items(), key=lambda item: int(item[0][1:]))
        for k, metrics in sorted_k_items:
            ks.append(int(k[1:]))  # 转换k值为整数
            fxs.append(metrics['f(x)'])  # 获取f(x)的值
        # 绘制折线图
        plt.plot(ks, fxs, label=iteration, marker='o')  # 绘制每个迭代的f(x)值变化
    # 添加图例和标签
    if int(iteration[9:]) < 15:
        plt.legend(title='Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('f(x)')
    plt.title(f'Update Type {type}: f(x) values across different iterations when n={n}')
    plt.grid(True)  # 添加网格线

    plt.savefig(f'figures/Week11_Problem2_{n}_{type}.png')
    plt.show()


def figure_draw_fxs(n, fxs_1, fxs_2, fxs_3):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.plot(fxs_1, label='Type 1', marker='o')
    plt.plot(fxs_2, label='Type 2', marker='+')
    plt.plot(fxs_3, label='Type 3', marker='*')
    # 添加图例和标签
    plt.legend(title='Update Type')
    plt.xlabel('Iterations')
    plt.ylabel('f(x)')
    plt.title(f'f(x) values across different iterations when n={n}')
    plt.grid(True)  # 添加网格线
    # 标注各点的数据值

    plt.savefig(f'figures/Week11_Problem2_{n}_fxs.png')
    plt.show()


def main():
    n_values = [100, 1000]
    precision = 0.001
    max_iter = 100
    results = {}
    for n in n_values:
        print('---------- n =', n, '----------')
        x = initial_vector(n).astype(dtype=np.float64)
        fx_1, iteration_1, datas_1, fxs_1 = BCD_type_1(x, precision, max_iter)
        fx_2, iteration_2, datas_2, fxs_2 = BCD_type_2(x, precision, max_iter)
        fx_3, iteration_3, datas_3, fxs_3 = BCD_type_3(x, precision, max_iter)
        figure_draw_datas(n, datas_1, 1)
        figure_draw_datas(n, datas_2, 2)
        figure_draw_datas(n, datas_3, 3)
        figure_draw_fxs(n, fxs_1, fxs_2, fxs_3)
        results[n] ={
            'Type 1': {'f(x)': fx_1, 'Iterations': iteration_1},
            'Type 2': {'f(x)': fx_2, 'Iterations': iteration_2},
            'Type 3': {'f(x)': fx_3, 'Iterations': iteration_3}
        }
    print('************* Results *************')
    print(json.dumps(results, indent=4, ensure_ascii=False))
    return results


# if __name__ == '__main__':
#     print('************* Student Info:周聪杰 2023320397 *************')
#     main()
