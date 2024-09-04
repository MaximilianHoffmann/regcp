from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy import interpolate 
import cupy as cp
import numpy as np
from tqdm import tqdm
import h5py

class RegCP:
    def __init__(self,vol: np.ndarray, smooth_sigma: float = 1.2, sig_map: float = 2,subsample: int = 1,bin_factor: int = 1,mem_limit: int = 12,n_avg: int = 10,do_line_corr:bool=True):
        """
        Initialize the RegCP (Registration with CuPy) object.

        This class is designed for registering volumetric image stacks using GPU acceleration.

        Parameters:
        -----------
        vol : np.ndarray
            Input volumetric image stack.
        smooth_sigma : float, optional
            Sigma for Gaussian smoothing of input images. Default is 1.2.
        sig_map : float, optional
            Sigma for Gaussian smoothing of correlation map. Default is 2.
        subsample : int, optional
            Subsampling factor for registration. Default is 1.
        bin_factor : int, optional
            Binning factor for registration. Default is 1.
        mem_limit : int, optional
            Memory limit in GB for GPU operations. Default is 12.
        n_avg : int, optional
            Number of images to average for reference image. Default is 10.
        do_line_corr : bool, optional
            Whether to perform line correction. Default is True.
        """
        self.vol = vol
        self.smooth_sigma = smooth_sigma
        self.sig_map = sig_map
        self.subsample = subsample
        self.bin_factor = bin_factor
        self.mem_limit = mem_limit
        self.ref_vol=None
        self.shifts=None
        self.do_line_corr=do_line_corr
        self.n_avg=n_avg
        self.line_corr_shift=np.zeros(self.vol.shape[1])
            
    def get_parameters(self) -> dict:
        """
        Returns the current parameters of the RegCP object as a dictionary.

        Returns:
        --------
        dict
            A dictionary containing the current parameter values.
        """
        return {
            'smooth_sigma': self.smooth_sigma,
            'sig_map': self.sig_map,
            'subsample': self.subsample,
            'bin_factor': self.bin_factor,
            'mem_limit': self.mem_limit,
            'n_avg': self.n_avg,
            'do_line_corr': self.do_line_corr,
            'shifts': self.shifts,
            'line_corr_shift': self.line_corr_shift,
            'ref_vol': self.ref_vol
        }
        
    def save_reg_run(self, filename: str):
        """
        Writes the current parameters to an HDF5 file.

        Parameters:
        -----------
        filename : str
            The name of the HDF5 file to write to.
        """
        params = self.get_parameters()
        with h5py.File(filename, 'w') as f:
            for key, value in params.items():
                if isinstance(value, (np.ndarray, list)):
                    f.create_dataset(key, data=value, compression="gzip")
                else:
                    f.attrs[key] = value
    
    def load_reg_run(self, filename: str):
        """
        Loads parameters from an HDF5 file and sets them as attributes of the RegCP object.

        Parameters:
        -----------
        filename : str
            The name of the HDF5 file to read from.
        """
        with h5py.File(filename, 'r') as f:
            for key in self.get_parameters().keys():
                if key in f:
                    setattr(self, key, f[key][()])
                elif key in f.attrs:
                    setattr(self, key, f.attrs[key])
                else:
                    print(f"Warning: '{key}' not found in the HDF5 file.")

      
    
    def build_reference_volume(self):
        """
        Constructs a reference volume for image registration by processing each plane of the input volume.

        This method iterates through each plane of self.vol, performing the following operations:
        1. Converts each plane to a CuPy array of type 'single'.
        2. Applies binning (factor: self.bin_factor) and subsampling (factor: self.subsample).
        3. Utilizes build_ref_image() to create a reference image for each plane.
        4. If self.do_line_corr is True, performs resonant scanning correction:
           - Calculates shift using resonant_scanning_correction().
           - Applies the shift to even lines of the reference image.
        5. Stores the processed plane in self.ref_vol.


        Modifies:
            self.ref_vol (np.ndarray): Initialized as zeros, shape matches a single plane of self.vol.
            self.line_corr_shift (np.ndarray): If line correction is enabled, stores shifts for each plane.

        Note:
            Assumes self.vol, self.bin_factor, self.subsample, self.do_line_corr, and self.n_avg are pre-defined.
            Requires sufficient memory to process individual planes.
        """
        self.ref_vol = np.zeros_like(self.vol[0].squeeze(),dtype=np.float32)
        for ii in tqdm(range(self.vol.shape[1])):
            plane=self.vol[:,ii,:,:]
            pl_c=cp.array(plane,'single')
            
            n_bin = self.bin_factor
            sub = self.subsample
            pl_b = cp.copy(pl_c[:(pl_c.shape[0]//n_bin)*n_bin].reshape(pl_c.shape[0]//n_bin, n_bin, pl_c.shape[1], pl_c.shape[2]).mean(1)[::sub])
            del pl_c
            cp._default_memory_pool.free_all_blocks()   
            ref_im=build_ref_image(pl_b, self.n_avg)
            if self.do_line_corr:
                shift=resonant_scanning_correction(ref_im,axis=1)
                self.line_corr_shift[ii]=shift
                ref_im[::2]=cp.roll(ref_im[::2],shift,1)
            self.ref_vol[ii]=ref_im.get()
            del pl_b,ref_im
            
   

    def determine_shifts(self):
        """
        Determine shifts between each plane of the input volume and the reference volume.

        This method iterates through each plane of self.vol, performing the following operations:
        1. Converts each plane to a CuPy array of type 'single'.
        2. Applies binning (factor: self.bin_factor) and subsampling (factor: self.subsample).
        3. Registers the processed plane with the reference image using register_plane().
        4. Stores the computed shifts in shift_list.

        Returns:
            shift_list (list): List of shift arrays, each corresponding to a plane in self.vol.
        """
        shift_list=[]
        for ii in tqdm(range(self.vol.shape[1]),desc="Determining shifts"):
            plane=self.vol[:,ii,:,:]
            pl_c=cp.array(plane,'single')
            
            n_bin = self.bin_factor
            sub = self.subsample
            shift_ind=cp.arange(0,pl_c.shape[0],n_bin)[::sub]
            pl_b = cp.copy(pl_c[:(pl_c.shape[0]//n_bin)*n_bin].reshape(pl_c.shape[0]//n_bin, n_bin, pl_c.shape[1], pl_c.shape[2]).mean(1)[::sub])
            del pl_c
            cp._default_memory_pool.free_all_blocks()   
            ref_im=cp.array(self.ref_vol[ii],'single')
            if self.do_line_corr:
                pl_b[:,::2,:]=cp.roll(pl_b[:,::2,:],self.line_corr_shift[ii],1)
            shifts,corr=register_plane(ref_im,pl_b)
            all_indices = cp.arange(self.vol.shape[0])
            shift_all = interpolate.interpn((shift_ind,),shifts.T,(all_indices,),method='nearest',bounds_error=False,fill_value=None)
            shift_list.append(shift_all.get())
            del shifts,corr,pl_b,ref_im
            cp._default_memory_pool.free_all_blocks()
        self.shifts=np.array(shift_list)
        return self.shifts
                        
    def apply_shifts(self,vol=None,max_shifts=20):
        shifts=self.shifts
        if vol is None:
            vol=self.vol
        ix=(np.abs(shifts)>max_shifts)
        shifts[ix]=0
        for ii,sh in enumerate(self.line_corr_shift):
            vol[:,ii,::2,:]=np.roll(self.vol[:,ii,::2,:],int(sh),axis=2)
        for iplane in range(vol.shape[1]):
            for ii in tqdm(range(vol.shape[0])):
                s=shifts[iplane][ii]
                vol[ii,iplane]=np.roll(np.roll(vol[ii,iplane],int(s[0]),axis=0),int(s[1]),axis=1)
        return vol
        

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

def build_ref_image(pl_b: cp.ndarray, n_avg: int = 10) -> cp.ndarray:
    """
    Build a reference image from a stack of images.

    Parameters:
    -----------
    pl_b : cp.ndarray
        Stack of images to be averaged.
    n_avg : int, optional
        Number of images to average. Default is 10.

    Returns:
    --------
    ref_im : cp.ndarray
        Reference image.
    """
    #TODO iteratively register images
    mid_image = pl_b[pl_b.shape[0] // 2]    
    pl_b_whitened = (pl_b - cp.mean(pl_b, axis=(1, 2))[:, None, None]) / cp.std(pl_b, axis=(1, 2))[:, None, None]
    mid_image_whitened = (mid_image - cp.mean(mid_image)) / cp.std(mid_image)
    correlations = cp.sum(pl_b_whitened * mid_image_whitened, axis=(1, 2))
    top_indices = cp.argsort(correlations)[-n_avg:]
    ref_im = cp.mean(pl_b[top_indices], axis=0)
    del pl_b_whitened, mid_image_whitened, correlations, top_indices
    return ref_im



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
    #TODO fix eps based in image size and amplitude
    pl_b_ft /= (cp.abs(pl_b_ft) + eps)
    ref_ft /= (cp.abs(ref_ft) + eps)
    corr = gaussian_filter(cp.fft.fftshift(cp.abs(cp.fft.ifft2(pl_b_ft * cp.conj(ref_ft))), axes=[1, 2]), [0, sig_map, sig_map])
    max_loc = cp.unravel_index(cp.argmax(corr, axis=(1, 2)), im_ref.shape)
    shift = -cp.array(max_loc) + cp.array(im_ref.shape)[:, None] / 2
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


@free_gpu_memory
def resonant_scanning_correction(img: cp.ndarray, axis=1) -> cp.ndarray:
    """
    Perform resonant scanning correction on an image.

    This function corrects for the distortion caused by resonant scanning in microscopy images.
    It calculates the shift between odd and even lines of the image using Fourier analysis.

    Parameters:
    -----------
    img : cp.ndarray
        Input image to be corrected. Should be a 2D CuPy array.
    axis : int, optional
        Axis along which to perform the correction. Default is 1 (columns).
        Use 0 for rows, 1 for columns.

    Returns:
    --------
    shift : float
        The calculated shift between odd and even lines of the image.
        This value can be used to correct the resonant scanning distortion.

    """
    if axis==0:
        ref_img=img.T
    else:
        ref_img=img

    ref_w=(ref_img-ref_img.mean(axis=1,keepdims=True))/ref_img.std(axis=1,keepdims=True)
    ref_ft=cp.fft.fft(ref_w,axis=1)
    odd=ref_ft[1::2,:]
    even=ref_ft[::2,:]    
    corr=cp.real(cp.fft.ifft(odd*cp.conj(even),axis=1))
    shift=cp.argmax(cp.fft.fftshift(corr.mean(0)))-ref_w.shape[1]//2
    return shift