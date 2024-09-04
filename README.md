# REGCP
## !! Work in progress !!
REGCP is a Python package for registering volumetric image stacks using GPU acceleration. Should eventually be a drop in replacement for suite2p registration.

Currently:
- Scanphase correction
- Phase correlation and correction of integer phase shifts
- Trivial reference volume: i.e. x highest correleated imgs without initial registration


TODO:
- Initial registration during generation of reference volume
- Implement GPU Memory Limit and chunked processing for large datasets
- Subpixel registration
- Non-rigid registration
- Suite2p like masking and temporal filtering of cross-correlation

Usage:

```
from siffpy import SiffReader
import regcp

sr=SiffReader(p_img_dset)
vol=sr.get_frames(frames=sr.im_params.flatten_by_timepoints(color_channel=0))
vol=vol.reshape([sr.im_params.array_shape[ii] for ii in [0,1,3,4]])

reg_run=regcp.RegCP(vol,bin_factor=2,subsample=1,n_avg=100)

reg_run.build_reference_volume()
shifts=reg_run.determine_shifts()
reg_run.save_reg_run(p_out)



reg_run_loaded=regcp.load_reg_run(p_out)
vol_reg=regcp.apply_shifts(vol,reg_run_loaded.shifts,reg_run_loaded.line_corr_shift,inplace=False)

```
Currently 17.2 s for a datset of size (39000, 5, 256, 128) and temporal binning of 2 on
a NVIDIA GeForce RTX 4090

## See Also


- [suite2p](https://github.com/MouseLand/suite2p)
