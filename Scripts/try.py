import torch
import xarray as xr


print(torch.cuda.is_available())     # should be True
print(torch.version.cuda)            # e.g. 12.4
print(torch.cuda.get_device_name(0))