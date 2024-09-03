from cupyx.scipy.ndimage import gaussian_filter
import cupy as cp
import numpy as np
from tqdm import tqdm

class RegCP:
    def __init__(self,vol: np.ndarray, smooth_sigma: float = 1.2, sig_map: float = 2,subsample: int = 1,bin_factor: int = 1,mem_limit: int = 12):
        self.vol = vol
        self.smooth_sigma = smooth_sigma
        self.sig_map = sig_map
        self.subsample = subsample
        self.bin_factor = bin_factor
        self.mem_limit = mem_limit


    def determine_shifts(self):
        shift_list=[]
        for ii in tqdm(range(self.vol.shape[1]),desc="Determining shifts"):
            plane=self.vol[:,ii,:,:]
            pl_c=cp.array(plane,'single')
            
            n_bin = self.bin_factor
            sub = self.subsample
            pl_b = cp.copy(pl_c[:(pl_c.shape[0]//n_bin)*n_bin].reshape(pl_c.shape[0]//n_bin, n_bin, pl_c.shape[1], pl_c.shape[2]).mean(1)[::sub])
            del pl_c
            cp._default_memory_pool.free_all_blocks()   
            ref_im=cp.copy(pl_b[pl_b.shape[0]//2])
            shifts,corr=register_plane(ref_im,pl_b)
            shift_list.append(shifts)
            del shifts,corr,pl_b,ref_im
            cp._default_memory_pool.free_all_blocks()
        return shift_list
                        

        
        

def free_gpu_memory(func):
    """
    A decorator that frees all GPU memory blocks after the decorated function is executed.

    This decorator wraps the given function and ensures that all GPU memory blocks
    are freed after the function's execution, helping to manage GPU memory usage.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: A wrapper function that executes the original function and then frees GPU memory.

    Example:
        @free_gpu_memory
        def some_gpu_intensive_function():
            # Function body
            pass
    """
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func


        



def gaussian_fft(smooth_sigma: float, Ly: int, Lx: int) -> cp.ndarray:
    """
    Creates a 2D Gaussian filter in the frequency domain.

    Parameters:
    -----------
    smooth_sigma : float
        Standard deviation of the Gaussian filter.
    Ly : int
        Height of the filter.
    Lx : int
        Width of the filter.

    Returns:
    --------
    filt : cp.ndarray
        2D array of shape (Ly, Lx) containing the Gaussian filter in the frequency domain.
    """
    # Create frequency grids
    fy = cp.fft.fftfreq(Ly).reshape(-1, 1)
    fx = cp.fft.fftfreq(Lx).reshape(1, -1)
    
    # Compute squared distance from origin in frequency space
    r2 = fy**2 + fx**2
    
    # Create Gaussian filter
    filt = cp.exp(-2 * (cp.pi**2) * (smooth_sigma**2) * r2)
    
    return filt.astype(cp.float32)



def register_plane(im_ref: cp.ndarray, pl_b: cp.ndarray, eps: float = 200, sig: float = 1.2, sig_map: float = 2) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute the shift between a reference image and a stack of images using phase correlation.

    Parameters:
    -----------
    im_ref : cp.ndarray
        Reference image.
    pl_b : cp.ndarray
        Stack of images to be aligned.
    eps : float, optional
        Small constant to avoid division by zero. Default is 200.
    sig : float, optional
        Sigma for Gaussian filtering of input images. Default is 1.2.
    sig_map : float, optional
        Sigma for Gaussian filtering of correlation map. Default is 2.

    Returns:
    --------
    shift : cp.ndarray
        Computed shifts for each image in pl_b relative to im_ref.
    corr : cp.ndarray
        Correlation map for each image in pl_b with im_ref.
    """
    pl_b_ft = cp.fft.fft2(gaussian_filter(pl_b, [0, sig, sig]))
    ref_ft = cp.fft.fft2(gaussian_filter(im_ref, [sig, sig]))
    pl_b_ft /= (cp.abs(pl_b_ft) + eps)
    ref_ft /= (cp.abs(ref_ft) + eps)
    corr = gaussian_filter(cp.fft.fftshift(cp.abs(cp.fft.ifft2(pl_b_ft * cp.conj(ref_ft))), axes=[1, 2]), [0, sig_map, sig_map])
    max_loc = cp.unravel_index(cp.argmax(corr, axis=(1, 2)), im_ref.shape)
    shift = cp.array(max_loc) - cp.array(im_ref.shape)[:, None] / 2
    del pl_b_ft, ref_ft
    return shift, corr


@free_gpu_memory
def apply_shifts(pl_b: cp.ndarray, shifts: cp.ndarray) -> cp.ndarray:
    """
    Apply 2D shifts to a stack of images efficiently without using a loop.

    Parameters:
    -----------
    pl_b : cp.ndarray
        Stack of images to be shifted.
    shifts : cp.ndarray
        Array of shifts to apply, shape (2, num_images).

    Returns:
    --------
    shifted_pl_b : cp.ndarray
        Shifted version of pl_b.
    """
    num_images = pl_b.shape[0]
    y, x = cp.meshgrid(cp.arange(pl_b.shape[1]), cp.arange(pl_b.shape[2]), indexing='ij')
    y_shifted = (y[None, :, :] - shifts[0, :, None, None]) % pl_b.shape[1]
    x_shifted = (x[None, :, :] - shifts[1, :, None, None]) % pl_b.shape[2]
    shifted_pl_b = pl_b[cp.arange(num_images)[:, None, None], y_shifted.astype(int), x_shifted.astype(int)]
    
    return shifted_pl_b
