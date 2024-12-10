import numpy as np

# Định nghĩa hàm mục tiêu
def objective_function(x):
    return x * np.cos(x) - 0.5 * x * np.cos(0.6 * x**2)

# Phân phối Levy Flight
def levy_flight(Lambda):
    sigma_u = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
               (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, 1)
    v = np.random.normal(0, sigma_v, 1)
    step = u / abs(v)**(1 / Lambda)
    return step

# Thuật toán Cuckoo Search để tìm Min
def cuckoo_search_min(n, pa, max_iter, bounds):
    # Khởi tạo
    nests = np.random.uniform(bounds[0], bounds[1], n)
    fitness = np.array([objective_function(x) for x in nests])
    best_nest = nests[np.argmin(fitness)]  # Tìm min thay vì max
    best_fitness = np.min(fitness)
    
    history = []

    for t in range(max_iter):
        # Levy Flight để tạo tổ mới
        new_nests = []
        for nest in nests:
            step_size = levy_flight(1.5)  # Lambda = 1.5
            new_nest = nest + step_size * (nest - best_nest)
            new_nest = np.clip(new_nest, bounds[0], bounds[1])
            new_nests.append(new_nest)
        new_nests = np.array(new_nests)
        
        # Đánh giá tổ mới
        new_fitness = np.array([objective_function(x) for x in new_nests])
        
        # Cập nhật tổ
        for i in range(n):
            if new_fitness[i] < fitness[i]:  # Tìm min
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
        
        # Thay thế tổ ngẫu nhiên với xác suất pa
        for i in range(n):
            if np.random.rand() < pa:
                nests[i] = np.random.uniform(bounds[0], bounds[1])
                fitness[i] = objective_function(nests[i])
        
        # Cập nhật tổ tốt nhất
        current_best = nests[np.argmin(fitness)]
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_nest = current_best
            best_fitness = current_best_fitness

        # Ghi lại lịch sử kết quả
        history.append((best_nest, best_fitness))
    
    return best_nest, best_fitness, history

# Thông số thuật toán
n = 15          # Số lượng tổ
pa = 0.25       # Xác suất phát hiện tổ
max_iter = 100  # Số vòng lặp
bounds = [5, 13]  # Miền giá trị

# Chạy thuật toán để tìm Min
best_x_min, best_fitness_min, history_min = cuckoo_search_min(n, pa, max_iter, bounds)

# Kết quả
print(f"Giá trị tối ưu x*: {best_x_min}")
print(f"Giá trị hàm tối ưu F(x*): {best_fitness_min}")