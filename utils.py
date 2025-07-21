import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter, label, center_of_mass



def mollifier(center, shape, particleSize, sigma_0):
    x, y = np.meshgrid(range(shape[0]), range(shape[1]))
    sigma = sigma_0 * particleSize * np.sqrt(2 / np.pi)
    scale = 1 / (2 * np.pi * sigma**2)
    arg = ((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2)
    fx0 = np.exp(-arg) * scale
    return fx0


def laplacian_of_gaussian(center, shape, particleSize, sigma_0):
    x, y = np.meshgrid(range(shape[0]), range(shape[1]))
    sigma = sigma_0 * particleSize * np.sqrt(2 / np.pi)
    scale = 1 / (2 * np.pi * sigma**2)
    arg = ((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2)
    fx0 = np.exp(-arg) * scale
    lap = -((x - center[0])**2 + (y - center[1])**2 - 2 * sigma**2) / (sigma**4) * fx0
    return lap


def get_p_local(fx_crop, fy_crop):
    mag = np.sqrt(fx_crop**2 + fy_crop**2)
    cy, cx = np.unravel_index(np.argmax(mag), mag.shape)
    fxhat, fyhat = -fx_crop[cy, cx], -fy_crop[cy, cx]
    norm = np.sqrt(fxhat**2 + fyhat**2)
    return np.array([fxhat / norm, fyhat / norm])

def get_center(fx,fy):
    cy, cx = np.unravel_index(np.argmax(fx**2 + fy**2), fx.shape)
    px , py = get_p_local(fx,fy)
    cy = round(cy - (4*py))
    cx = round(cx - (4*px))

    return cx , cy


def vector_to_angle(vector):
    return np.arctan2(vector[1], vector[0]) * 180 / np.pi


def detect_particles(fx, fy, threshold_ratio=0.5, min_distance=40):
    mag = np.sqrt(fx**2 + fy**2)
    mag_filtered = maximum_filter(mag, size=min_distance)
    peaks = (mag == mag_filtered) & (mag > threshold_ratio * np.max(mag))
    labeled, num_features = label(peaks)
    centers = center_of_mass(peaks, labeled, range(1, num_features + 1))
    centers = [(int(c[1]), int(c[0])) for c in centers]  # convert to (x, y)
    return centers


def estimate_single_particle(fx, fy, particleSize, center):
    min_error = float('inf')
    best_params = None
    best_center = None
    best_p = None

    cx, cy = center
    delta_a = 0.01 * particleSize

    # if cx < 18:
    #     cx += 4
    # if cx > fx.shape[0] - 18:
    #     cx -= 4
    # if cy < 18:
    #     cy += 4
    # if cy > fx.shape[1] - 18:
    #     cy -= 4

    try:
        crop_fx = fx[cy - 15:cy + 16, cx - 15:cx + 16]
        crop_fy = fy[cy - 15:cy + 16, cx - 15:cx + 16]
        p = get_p_local(crop_fx, crop_fy)
    except:
        return {'f0': None, 'd0': None, 'error': np.inf, 'center': None, 'p': None}
    
    cx, cy = get_center(crop_fx, crop_fy)
    cx += center[0] - 15
    cy += center[1] - 15

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            cx_new, cy_new = cx + dx, cy + dy
            try:
                laplacian = laplacian_of_gaussian((cx_new, cy_new), fx.shape, particleSize, sigma_0=1)
                rp = (cx_new + delta_a * p[0], cy_new + delta_a * p[1])
                rn = (cx_new - delta_a * p[0], cy_new - delta_a * p[1])
                mp = mollifier(rp, fx.shape, particleSize, sigma_0=1)
                mn = mollifier(rn, fx.shape, particleSize, sigma_0=1)

                fxmask = fx[cy_new - 18:cy_new + 19, cx_new - 18:cx_new + 19]
                fymask = fy[cy_new - 18:cy_new + 19, cx_new - 18:cx_new + 19]
                mpmask = mp[cy_new - 18:cy_new + 19, cx_new - 18:cx_new + 19]
                mnmask = mn[cy_new - 18:cy_new + 19, cx_new - 18:cx_new + 19]
                laplacianmask = laplacian[cy_new - 18:cy_new + 19, cx_new - 18:cx_new + 19]

                def objective(params):
                    f0, d0 = params
                    fx_hat = -f0 * mpmask * p[0] + f0 * mnmask * p[0] - d0 * laplacianmask * p[0]
                    fy_hat = -f0 * mpmask * p[1] + f0 * mnmask * p[1] - d0 * laplacianmask * p[1]
                    return np.sum((fxmask - fx_hat)**2 + (fymask - fy_hat)**2)

                result = minimize(objective, [25000, 25000], bounds=[(10000, 60000), (10000, 60000)])

                if result.fun < min_error:
                    min_error = result.fun
                    best_params = result.x
                    best_center = (cx_new, cy_new)
                    best_p = p

            except:
                continue

    return {
        'f0': best_params[0],
        'd0': best_params[1],
        'error': min_error,
        'center': best_center,
        'p': best_p
    }


# def estimate_all_particles(fx, fy, particleSize):
#     centers = detect_particles(fx, fy)
#     particles_info = []

#     for center in centers:
#         info = estimate_single_particle(fx, fy, particleSize, center)
#         if info['center'] is not None:
#             particles_info.append(info)

#     return particles_info

def estimate_all_particles(fx, fy, particleSize):
    Nx, Ny = fx.shape
    cx_mid, cy_mid = Nx // 2, Ny // 2  # Midpoint of the image

    centers = detect_particles(fx, fy)
    particles_info = []

    for center in centers:
        info = estimate_single_particle(fx, fy, particleSize, center)
        if info['center'] is not None:
            particles_info.append(info)

    def get_quadrant(center):
        cx, cy = center
        if cx < cx_mid and cy < cy_mid:
            return 1  # Top-left
        elif cx < cx_mid and cy >= cy_mid:
            return 2  # Bottom-left
        elif cx >= cx_mid and cy < cy_mid:
            return 3  # Top-right
        else:
            return 4  # Bottom-right

    # Sort particles by quadrant
    particles_info.sort(key=lambda info: get_quadrant(info['center']))

    return particles_info


def vel_from_force(fx, fy, particleSize, sigma_0):
    Nx, Ny = fx.shape
    x, y = np.meshgrid(range(Nx), range(Ny))
    kx_ = (2 * np.pi / 1) * np.fft.fftfreq(Nx)
    ky_ = (2 * np.pi / 1) * np.fft.fftfreq(Ny)
    kx, ky = np.meshgrid(kx_, ky_)

    Fxk = np.fft.fft2(fx)
    Fyk = np.fft.fft2(fy)

    k2 = kx * kx + ky * ky
    ik2 = np.zeros_like(k2)
    np.divide(1, k2, out=ik2, where=(k2 > 1e-6))

    Fdotk = Fxk * kx + Fyk * ky
    vxk = (Fxk - Fdotk * (kx * ik2)) * ik2
    vyk = (Fyk - Fdotk * (ky * ik2)) * ik2
    vxk[0, 0] = 0
    vyk[0, 0] = 0

    vx = np.real(np.fft.ifft2(vxk))
    vy = np.real(np.fft.ifft2(vyk))

    return vx, vy
