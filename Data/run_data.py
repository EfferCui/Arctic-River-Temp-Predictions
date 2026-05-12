import xarray as xr
#import matplotlib.plot as plt
ds = xr.open_dataset(r"c:\Users\Effer\Arctic-River-Temp-Predictions-main\Arctic-River-Temp-Predictions-main\Data\1_test\time_series\3.nc")

print(ds)
#get variables
print(ds.data_vars)
#get actual variable values
mean_temp_c= ds["temperature_2m_mean"].values





#get coordinates
print(ds.coords)
#get actual coordinates values
coordinates= ds.coords["date"].values

#%%
#convert xarray to dataframe
df = ds.to_dataframe()