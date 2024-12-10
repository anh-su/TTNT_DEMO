import numpy as np
import matplotlib.pyplot as plt

# Hàm tính cường độ tín hiệu
def tinh_cuong_do_tin_hieu(vi_tri, diem_phu_song):
    cuong_do = 0
    for vi_tri_tram in vi_tri:
        khoang_cach = np.linalg.norm(diem_phu_song - vi_tri_tram, axis=1)
        cuong_do += np.sum(1 / (khoang_cach + 1e-6))  # Tránh chia cho 0
    return cuong_do

# Hàm tính chi phí khoảng cách
def chi_phi_khoang_cach(vi_tri):
    tong_chi_phi = 0
    for i in range(len(vi_tri)):
        for j in range(i + 1, len(vi_tri)):
            tong_chi_phi += np.linalg.norm(vi_tri[i] - vi_tri[j])
    return tong_chi_phi

# Hàm mục tiêu
def ham_muc_tieu(vi_tri, diem_phu_song, trong_so=1.0):
    return -tinh_cuong_do_tin_hieu(vi_tri, diem_phu_song) + trong_so * chi_phi_khoang_cach(vi_tri)

# Thuật toán Firefly
def giai_thuat_firefly(so_dom_dom=20, so_tram=5, so_chieu=2, so_the_he=100, gioi_han=(0, 100), trong_so=1.0):
    # Khởi tạo vị trí ngẫu nhiên cho đom đóm
    dom_dom = [np.random.uniform(gioi_han[0], gioi_han[1], (so_tram, so_chieu)) for _ in range(so_dom_dom)]
    diem_phu_song = np.random.uniform(gioi_han[0], gioi_han[1], (50, so_chieu))  # Các điểm cần phủ sóng
    do_sang = [ham_muc_tieu(dd, diem_phu_song, trong_so) for dd in dom_dom]

    vi_tri_tot_nhat = None
    do_sang_tot_nhat = float('inf')

    for the_he in range(so_the_he):
        for i in range(so_dom_dom):
            for j in range(so_dom_dom):
                if do_sang[j] < do_sang[i]:  # Đom đóm j "sáng" hơn
                    khoang_cach = np.linalg.norm(dom_dom[i] - dom_dom[j])
                    beta = np.exp(-khoang_cach**2)
                    dom_dom[i] += beta * (dom_dom[j] - dom_dom[i]) + 0.2 * np.random.uniform(-0.5, 0.5, (so_tram, so_chieu))
                    dom_dom[i] = np.clip(dom_dom[i], gioi_han[0], gioi_han[1])  # Giữ trong giới hạn
                    do_sang[i] = ham_muc_tieu(dom_dom[i], diem_phu_song, trong_so)

        min_idx = np.argmin(do_sang)
        if do_sang[min_idx] < do_sang_tot_nhat:
            do_sang_tot_nhat = do_sang[min_idx]
            vi_tri_tot_nhat = dom_dom[min_idx]

        print(f"Thế hệ {the_he+1}: Độ sáng tốt nhất = {do_sang_tot_nhat}")

    return vi_tri_tot_nhat, do_sang_tot_nhat, diem_phu_song

# Chạy thuật toán
vi_tri_tot_nhat, do_sang_tot_nhat, diem_phu_song = giai_thuat_firefly()

print("\nVị trí trạm Wi-Fi tốt nhất tìm được:\n", vi_tri_tot_nhat)
print("Giá trị hàm mục tiêu tốt nhất:", do_sang_tot_nhat)

# Vẽ kết quả
def ve_ket_qua(vi_tri_tot_nhat, diem_phu_song):
    plt.scatter(diem_phu_song[:, 0], diem_phu_song[:, 1], c='blue', label='Điểm cần phủ sóng')
    plt.scatter(vi_tri_tot_nhat[:, 0], vi_tri_tot_nhat[:, 1], c='red', label='Trạm Wi-Fi', marker='x')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X (mét)')
    plt.ylabel('Y (mét)')
    plt.title('Vị trí tối ưu của các trạm Wi-Fi')
    plt.legend()
    plt.grid()
    plt.show()

ve_ket_qua(vi_tri_tot_nhat, diem_phu_song)
