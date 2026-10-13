# This module is for the REMO2020-TEB paper plotting
# J-PP 2026
#

import glob
import numpy as np
import xarray as xr
import xesmf as xe
import pyremo as pr
import cf_xarray as cfxr
from pathlib import Path
import cftime
import pandas as pd
import matplotlib
from pprint import pprint
import importlib
import os
import re
import sys

############################################################################################
############################################################################################
# define domain routine
def get_domain(domain):
    ds = pr.remo_domain(domain, add_vertices=True)
    return ds
############################################################################################
############################################################################################
# define E-OBS reading routine
def eobs_wind_open(var,eobsfile,ystart,yend,trggrid,dlim,lake_mask,lakemasklim):
    # Create target grid
    target_grid = get_domain(trggrid)
    # open E-OBS data
    eobs_daydata = xr.open_dataset(eobsfile, chunks={"time": 1, "latitude": 201, "longitude": 464}, use_cftime=False).sel(time=slice(str(ystart), str(yend)))
    # Chunk data
    eobs_daydatac = eobs_daydata.rename({'longitude': 'lon','latitude': 'lat'}).chunk(dict(lat=-1)).chunk(dict(lon=-1))
    # regrid data
    regridder = xe.Regridder(eobs_daydatac, target_grid, "bilinear")
    eobs_daydata_gridded = regridder(eobs_daydatac)
    eobs_daydata_gridded = eobs_daydata_gridded.where(eobs_daydata_gridded > 0, drop=False) # get rid off 0 zeros from regridding
    #
    eobs_mdata = (eobs_daydata_gridded.resample(time='1MS').mean()[var]).compute()
    #
    # mask months less than dlim days
    eobs_mask_tmp = xr.where(eobs_daydata_gridded[var] > 0.0, 1, 0)
    eobs_mask = xr.where(eobs_mask_tmp.resample(time='1MS').sum() >= dlim, 1, 0).rename("mask") # at least dlim days per gridbox per month
    # create final mask using lake mask
    final_mask_tmp = xr.merge([eobs_mask, lake_mask.squeeze().rename("lake_mask")], compat="override", join="override")
    final_mask = xr.where((final_mask_tmp.mask > 0.5) & (final_mask_tmp.lake_mask < lakemasklim),1,0).compute()
    # clean memory
    del eobs_daydata, eobs_daydatac, eobs_daydata_gridded, eobs_mask_tmp, eobs_mask
    #
    return eobs_mdata, final_mask
############################################################################################
############################################################################################
def urclim_mask_creator(inputoutput_dir,city,lon_city,lat_city,model,scenario,domain,urban_th,urban_sur_th,orog_diff,sftlf_th,ratio_r2u,min_city_size,lon_lim,lat_lim,res,urb_fracs):
    """ Create urban masks for different sources, based on urclim"""
    from urclimask.urban_areas import (
    UrbanVicinity
    )
    # create base filename
    base_filename = f'{city}-{domain}_urclim_masks'
    # Create output file name 
    foutput = f"{inputoutput_dir}/urmask_{base_filename}_fx.nc"
    # Load static variables, such as urban fraction (sfturf), terrain elevation (orography) or land-sea fraction (sftlf).
    file_sfturf = f"urban_fractions_{domain}.nc"
    file_orog = f"FIB_{domain}.nc"
    file_sftlf = f"BLA_{domain}.nc"
    # urban fraction done in urban loop
    ds_orog = xr.open_dataset(inputoutput_dir+file_orog).squeeze().drop_vars('time').rename({"FIB":"orog"})
    ds_sftlf = xr.open_dataset(inputoutput_dir+file_sftlf).drop_vars('time').rename({"BLA":"sftlf"}).squeeze()*100.0 # fraction to %
    # Loop over the different urban fractions and create output
    ii = 0
    for ufr in urb_fracs:
        # urban name
        urban_var = ufr
        # load urban fraction
        ds_sfturf = xr.open_dataset(inputoutput_dir+file_sfturf)[[f'{urban_var}']].squeeze().drop_vars('time')*100.0 # fraction to %
        # Add parameters to the function
        URBAN = UrbanVicinity(
            urban_sur_th = urban_sur_th,
            orog_diff = orog_diff,
            sftlf_th = sftlf_th,
            ratio_r2u = ratio_r2u,
            min_city_size = min_city_size,
            lon_city = lon_city,
            lat_city = lat_city,
            lon_lim = lon_lim,
            lat_lim = lat_lim,
            model = model,
            domain = domain,
            urban_th = urban_th,
            urban_var = urban_var
        )
        # create first one full domain with nans
        ds_sfturf_nan = ds_sfturf*np.nan
        # crop around the city
        ds_sfturf = URBAN.crop_area_city(ds = ds_sfturf, res = res)
        ds_orog = URBAN.crop_area_city(ds = ds_orog, res = res)
        ds_sftlf = URBAN.crop_area_city(ds = ds_sftlf, res = res)
        # Define masks using the parameters above
        ds_merged = xr.merge([ds_sfturf,ds_orog,ds_sftlf], compat="override", join="override")
        ds_sfturf = ds_merged[[f'{urban_var}']]
        ds_orog = ds_merged[["orog"]]
        ds_sftlf = ds_merged[["sftlf"]]
        del ds_merged
        sfturf_mask, sfturf_sur_mask, orog_mask, sftlf_mask = URBAN.define_masks(
            ds_sfturf = ds_sfturf, 
            ds_orog = ds_orog, 
            ds_sftlf = ds_sftlf
        )
        # Define rural vicinity areas (We use an iterative algorithm to select the urban vicinity excluding masked areas)
        urmask = URBAN.select_urban_vicinity(
            sfturf_mask = sfturf_mask, 
            orog_mask = orog_mask,
            sftlf_mask = sftlf_mask,
            sfturf_sur_mask = sfturf_sur_mask
        )
        # Update nan field with mask and rename
        ds_sfturf_nan.update(urmask.rename({"urmask":f'{urban_var}'}))
        # append if needed
        if(ii==0):
            ds_out = ds_sfturf_nan
        else:
            ds_out = xr.merge([ds_out,ds_sfturf_nan],compat="override", join="override")
        # increase counter
        ii = ii + 1
    # write out data
    ds_out.to_netcdf(foutput,mode="w")
############################################################################################
############################################################################################
def flip(items, ncol):
    """
    Flips the order of items in a list by grouping them into columns.
    *used to fill columns of legends first*

    Parameters:
        items (list): The list of items to be flipped.
        ncol (int): The number of columns to group the items into.

    Returns:
        itertools.chain: A chain object containing the flipped items.

    Source:
        https://stackoverflow.com/questions/10101141/
        matplotlib-legend-add-items-across-columns-instead-of-down
        https://github.com/matplotlib/matplotlib/issues/27067
    """
    import itertools
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
############################################################################################
############################################################################################
def parse_paris_time(ds):
    # Extract the TIME variable
    time_var = ds["TIME"]

    # Convert to datetime64[ns]
    base_time = pd.to_datetime("1970-01-01")
    datetime_array = base_time + pd.to_timedelta(time_var.values, unit="m")
    datetime64_array = np.array(datetime_array, dtype="datetime64[ns]")

    # Assign back to the dataset
    ds = ds.assign_coords(time=("time", datetime64_array))
    
    # Add 'season' and 'hour' as coordinates
    ds = ds.assign_coords(
        season=ds.time.dt.season,
        hour=ds.time.dt.hour
    )

    return ds
############################################################################################
############################################################################################
def get_paris_var_index(dsin,varname,indin,make_mean=False,addc=0):
    """ Read from Paris data a variable based on indexes"""
    # 
    dsout = dsin[varname].isel(space=indin)
    if(make_mean):
        dsout = dsout.mean(dim="space")+addc
    return dsout
############################################################################################
############################################################################################
# create variable info class for model
class parisclass:
    def __init__(self, varind, sitename, lat, lon, vartype):
        self.varind = varind # index of variable
        self.sitename = sitename # site name
        self.lat = lat # latitude
        self.lon = lon # longitude
        self.vartype = vartype # urban/crops/forest    
############################################################################################
############################################################################################
# create variable info class for model
class modelclass_cities:
    def __init__(self,userexp,runame,maskname,plstyle,resolution):
        self.userexp = userexp # XXXYYY
        self.runame = runame # name of the run
        self.maskname = maskname
        self.plstyle = plstyle
        self.reso = resolution
        self.rawdata = []
        self.vardata = []
        self.rawuhi = []
        self.urb_data = []
        self.rur_data = []
############################################################################################
############################################################################################
def prepare_remo_to_umask_seas(dsin,maskname,varname,lreplace_median,quantiles,luhi=False):
    """ Calculate """
    weight = np.cos(np.deg2rad(dsin.rlat)).compute()
    urban_tmp_raw = dsin[[varname]].where(dsin[maskname] > 0.5,drop=True).weighted(weight).mean(dim=("rlat", "rlon")).squeeze() # mask is 1.0
    rural_tmp_raw = dsin[[varname]].where(dsin[maskname] < 0.5,drop=True).weighted(weight).mean(dim=("rlat", "rlon")).squeeze()  # mask is 0.0
    urban_tmp = urban_tmp_raw.groupby(["season", "hour"]).quantile(quantiles, dim='time').rename({varname:"urban"}) # mask is 1.0
    rural_tmp = rural_tmp_raw.groupby(["season", "hour"]).quantile(quantiles, dim='time').rename({varname:"rural"})  # mask is 0.0
    if(luhi):
        uhi_tmp_raw = (urban_tmp_raw-rural_tmp_raw)
        uhi_tmp = uhi_tmp_raw.groupby(["season", "hour"]).quantile(quantiles, dim='time').rename({varname:"uhi"})
    if (lreplace_median):
        urban_tmp.loc[dict(quantile=0.5)] = urban_tmp_raw.groupby(["season", "hour"]).mean("time").rename({varname:"urban"}) # mask is 1.0
        rural_tmp.loc[dict(quantile=0.5)] = rural_tmp_raw.groupby(["season", "hour"]).mean("time").rename({varname:"rural"}) # mask is 1.0
        if(luhi):
            uhi_tmp.loc[dict(quantile=0.5)] = uhi_tmp_raw.groupby(["season", "hour"]).mean("time").rename({varname:"uhi"})
    #
    if(luhi):
        outds = xr.merge([urban_tmp, rural_tmp, uhi_tmp], compat="override", join="override")
    else:
        outds = xr.merge([urban_tmp, rural_tmp], compat="override", join="override")
    return outds#, uhi_tmp_raw
############################################################################################
############################################################################################
def badc_csv_to_netcdf(data_dir: str, input_filename: str):
    """
    Convert a BADC-CSV (MIDAS Open hourly obs) file to a NetCDF file with xarray.
    Drops rows with invalid or missing time (NaT), sorts by time, and removes duplicates.
    """

    def smart_split(line: str):
        parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line.strip())
        return [p.strip().strip('"') for p in parts]

    # ---------- 1) Read file and find 'data' section ----------
    in_path = os.path.join(data_dir, input_filename)
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_line_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "data":
            data_line_idx = i
            break
    if data_line_idx is None:
        raise ValueError("No 'data' section found in the file.")

    metadata_lines = lines[:data_line_idx]
    header_and_data_start = data_line_idx + 1

    # ---------- 2) Parse metadata ----------
    global_attrs = {}
    var_attrs = {}
    lat = lon = None
    height_val = None
    height_unit = None

    for raw in metadata_lines:
        parts = smart_split(raw)
        if not parts or len(parts) < 2:
            continue

        key = parts[0]
        scope = parts[1]

        if scope == "G":
            values = parts[2:]
            if key == "location" and len(values) >= 2:
                try:
                    lat = float(values[0])
                    lon = float(values[1])
                except ValueError:
                    lat, lon = values[0], values[1]
            elif key == "height":
                if values:
                    try:
                        height_val = float(values[0])
                    except ValueError:
                        height_val = values[0]
                if len(values) > 1:
                    height_unit = values[1]
                global_attrs["height"] = ", ".join(values)
            else:
                global_attrs[key] = ", ".join(values)
        else:
            if len(parts) < 3:
                continue
            var = parts[1]
            values = parts[2:]
            if var not in var_attrs:
                var_attrs[var] = {}
            if key == "long_name":
                var_attrs[var]["long_name"] = values[0]
                if len(values) > 1:
                    var_attrs[var]["units"] = values[1]
            elif key == "type":
                var_attrs[var]["original_type"] = values[0]
            elif key == "comments":
                prev = var_attrs[var].get("comment")
                add = ", ".join(values)
                var_attrs[var]["comment"] = f"{prev}; {add}" if prev else add

    # ---------- 3) Read data ----------
    df = pd.read_csv(
        in_path,
        skiprows=header_and_data_start,
        na_values=["NA"],
        dtype=None,
        low_memory=False,
    )

    # Convert time and clean
    df["ob_time"] = pd.to_datetime(df["ob_time"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["ob_time"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with invalid time values (NaT).")

    # Sort by time and remove duplicates
    df = df.sort_values("ob_time").drop_duplicates(subset=["ob_time"])
    print(f"Data sorted by time. Remaining rows: {len(df)}")

    # ---------- 4) Build Dataset ----------
    ds = xr.Dataset(coords={"time": ("time", df["ob_time"].values)})

    if lat is not None:
        ds.coords["latitude"] = lat
        ds["latitude"].attrs.update({"units": "degrees_north"})
    if lon is not None:
        ds.coords["longitude"] = lon
        ds["longitude"].attrs.update({"units": "degrees_east"})
    if height_val is not None:
        ds.coords["height"] = height_val
        ds["height"].attrs.update({"units": height_unit or "m"})

    ds.attrs.update(global_attrs)
    ds.attrs["featureType"] = global_attrs.get("feature_type", "timeSeries")

    for col in df.columns:
        if col == "ob_time":
            if col in var_attrs:
                ds["time"].attrs.update(var_attrs[col])
            continue
        ds[col] = ("time", df[col].values)
        if col in var_attrs:
            ds[col].attrs.update(var_attrs[col])

    ds.attrs["missing_value"] = "NA"

    if "id" in df.columns and df["id"].nunique() == 1:
        ds.coords["station_id"] = df["id"].iloc[0]
        ds["station_id"].attrs["long_name"] = var_attrs.get("id", {}).get("long_name", "station identifier")

    #out_path = os.path.join(data_dir, output_filename)
    #ds.to_netcdf(out_path)
    #print(f"NetCDF saved to {out_path}")
    return ds
############################################################################################
############################################################################################
def prepare_midas_data(midas_datapath,midas_filename,variable,quantiles,lreplace_median,addc=0):
    midas_in = badc_csv_to_netcdf(midas_datapath,midas_filename)
    midas_in_raw = midas_in.assign_coords(
        season=midas_in.time.dt.season,
        hour=midas_in.time.dt.hour
    )[[variable]]
    midas_out = midas_in_raw.groupby(["season", "hour"]).quantile(quantiles, dim='time')
    if (lreplace_median):
        midas_out.loc[dict(quantile=0.5)] = midas_in_raw.groupby(["season", "hour"]).mean("time")
    return (midas_out[variable]+addc, midas_in_raw)
############################################################################################
############################################################################################
def dwd_txt_to_netcdf(datapath, io_file, variables):
    """
    Convert a semicolon-delimited TXT file to a NetCDF file using xarray.

    Parameters
    ----------
    datapath : str
        Path to input/output files.
    io_file : str
        Name of the input/output file without suffix (added automatically).
    variables : list of str
        List of variable names (columns) to include in the NetCDF file.
    """
    # === STEP 1: READ THE TXT FILE ===
    df = pd.read_csv(datapath+io_file+'.txt', sep=';', skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]

    # === STEP 2: CLEAN DATA ===
    df.replace(-999, np.nan, inplace=True)
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d%H', errors='coerce')
    df.sort_values('MESS_DATUM', inplace=True)

    # === STEP 3: Create xarray Dataset ===
    time = df['MESS_DATUM'].values

    # Data variables (only selected ones)
    data_vars = {}
    for var in variables:
        if var in df.columns:
            data_vars[var] = (['time'], df[var].values)
        else:
            print(f"Warning: Variable '{var}' not found in the file and will be skipped.")

    # Coordinates
    coords = {'time': time}

    # Create Dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add attributes
    ds.attrs['title'] = 'DWD Meteorological Data'
    ds.attrs['source'] = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    ds.attrs['history'] = 'Converted from TXT to NetCDF using xarray'
    if 'STATIONS_ID' in df.columns:
        ds.attrs['station_id'] = str(df['STATIONS_ID'].values[0])

    # === STEP 4: SAVE TO NETCDF === (can be un-commented if needed)
    #ds.to_netcdf(datapath+io_file+'.nc', mode='w')
    #print(f"✅ NetCDF file created: {datapath+io_file+'.nc'}")
    return ds
############################################################################################
############################################################################################
def open_prepare_dwd(dpath,filename,varin,year,quantiles,lreplace_median,addc=0.0):
    """ Opend and prepare dwd data"""
    #dwd_raw = xr.open_dataset(dpath+filename)[[varin]].squeeze()+addc
    dwd_raw = dwd_txt_to_netcdf(dpath,filename,[varin])+addc
    dwd_raw = dwd_raw.where(dwd_raw['time'].dt.year == year, drop=True)
    dwd_raw = dwd_raw[{varin}].assign_coords(
        season=dwd_raw[{varin}].time.dt.season,
        hour=dwd_raw[{varin}].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_data = dwd_raw.quantile(quantiles, dim='time')[varin]
    if (lreplace_median):
        dwd_data.loc[dict(quantile=0.5)] = dwd_raw.mean("time")[varin]
    return dwd_data.compute()
############################################################################################
############################################################################################
def open_prepare_dwd_uhi_new(dpath,uhivar,urfiles,urvar,rurfiles,rurvarin,year,quantiles,lreplace_median,addc=0.0):
    """ Opend and prepare dwd data"""
    #dwd_raw = xr.open_dataset(dpath+filename)[[varin]].squeeze()+addc
    datasets = []
    for rfile in urfiles:
        if rfile.startswith("produkt_air_temperature_"):
            urvarin = urvar
        elif rfile.startswith("produkt_tu_stunde_"):
            urvarin = rurvarin
        rds = dwd_txt_to_netcdf(dpath,rfile,[urvarin])+addc
        rds = rds.where(rds['time'].dt.year == year, drop=True)
        datasets.append(rds)
        del rds
    rds_all = xr.concat(
        datasets,
        dim="station",
        join="outer" # keep all timestamps from all files
    )
    dwd_raw_urb = rds_all.mean(dim="station", skipna=True)
    del rds_all
    # prepare rural files and make a mean timewise
    datasets = []
    for rfile in rurfiles:
        rds = dwd_txt_to_netcdf(dpath,rfile,[rurvarin])+addc
        rds = rds.where(rds['time'].dt.year == year, drop=True).rename({rurvarin:urvarin})
        datasets.append(rds)
        del rds
    rds_all = xr.concat(
        datasets,
        dim="station",
        join="outer" # keep all timestamps from all files
    )
    dwd_raw_rur = rds_all.mean(dim="station", skipna=True)
    del rds_all
    dwd_raw_uhi = (dwd_raw_urb - dwd_raw_rur).rename({urvarin:uhivar})
    dwd_raw_uhi = dwd_raw_uhi[uhivar].assign_coords(
        season=dwd_raw_uhi[uhivar].time.dt.season,
        hour=dwd_raw_uhi[uhivar].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_uhi = dwd_raw_uhi.quantile(quantiles, dim='time')
    dwd_raw_urb = dwd_raw_urb.assign_coords(
        season=dwd_raw_urb[urvarin].time.dt.season,
        hour=dwd_raw_urb[urvarin].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_raw_urb_q = dwd_raw_urb.quantile(quantiles, dim='time')
    dwd_raw_rur = dwd_raw_rur.assign_coords(
        season=dwd_raw_rur[urvarin].time.dt.season,
        hour=dwd_raw_rur[urvarin].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_raw_rur_q = dwd_raw_rur.quantile(quantiles, dim='time')
    if (lreplace_median):
        dwd_uhi.loc[dict(quantile=0.5)] = dwd_raw_uhi.mean("time")
        dwd_raw_urb_q.loc[dict(quantile=0.5)] = dwd_raw_urb.mean("time")
        dwd_raw_rur_q.loc[dict(quantile=0.5)] = dwd_raw_rur.mean("time")
    return dwd_uhi.compute(), dwd_raw_urb_q[urvarin].squeeze().compute(), dwd_raw_rur_q[urvarin].squeeze().compute()
############################################################################################
############################################################################################
def open_prepare_dwd_uhi(dpath,uhivar,urfile,urvarin,rurfiles,rurvarin,year,quantiles,lreplace_median,addc=0.0):
    """ Opend and prepare dwd data"""
    #dwd_raw = xr.open_dataset(dpath+filename)[[varin]].squeeze()+addc
    dwd_raw_urb = dwd_txt_to_netcdf(dpath,urfile,[urvarin])+addc
    dwd_raw_urb = dwd_raw_urb.where(dwd_raw_urb['time'].dt.year == year, drop=True)
    # prepare rural files and make a mean timewise
    datasets = []
    for rfile in rurfiles:
        rds = dwd_txt_to_netcdf(dpath,rfile,[rurvarin])+addc
        rds = rds.where(rds['time'].dt.year == year, drop=True).rename({rurvarin:urvarin})
        datasets.append(rds)
        del rds
    rds_all = xr.concat(
        datasets,
        dim="station",
        join="outer" # keep all timestamps from all files
    )
    dwd_raw_rur = rds_all.mean(dim="station", skipna=True)
    del rds_all
    dwd_raw_uhi = (dwd_raw_urb - dwd_raw_rur).rename({urvarin:uhivar})
    dwd_raw_uhi = dwd_raw_uhi[uhivar].assign_coords(
        season=dwd_raw_uhi[uhivar].time.dt.season,
        hour=dwd_raw_uhi[uhivar].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_uhi = dwd_raw_uhi.quantile(quantiles, dim='time')
    dwd_raw_urb = dwd_raw_urb.assign_coords(
        season=dwd_raw_urb[urvarin].time.dt.season,
        hour=dwd_raw_urb[urvarin].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_raw_urb_q = dwd_raw_urb.quantile(quantiles, dim='time')
    dwd_raw_rur = dwd_raw_rur.assign_coords(
        season=dwd_raw_rur[urvarin].time.dt.season,
        hour=dwd_raw_rur[urvarin].time.dt.hour
    ).groupby(["season", "hour"])
    dwd_raw_rur_q = dwd_raw_rur.quantile(quantiles, dim='time')
    if (lreplace_median):
        dwd_uhi.loc[dict(quantile=0.5)] = dwd_raw_uhi.mean("time")
        dwd_raw_urb_q.loc[dict(quantile=0.5)] = dwd_raw_urb.mean("time")
        dwd_raw_rur_q.loc[dict(quantile=0.5)] = dwd_raw_rur.mean("time")
    return dwd_uhi.compute(), dwd_raw_urb_q[urvarin].squeeze().compute(), dwd_raw_rur_q[urvarin].squeeze().compute()
############################################################################################
############################################################################################
def HH_to_netcdf_multi(filepath, prefix):
    """
    Convert multiple semicolon-separated meteorological TXT files to one NetCDF with metadata.
    
    Parameters:
    filepath (str): Directory containing the TXT files
    outputname (str): Name of the output NetCDF file
    """
    from glob import glob
    
    # Find all TXT files in the directory
    files = sorted(glob(os.path.join(filepath, prefix+"*.txt")))
    if not files:
        raise FileNotFoundError("No TXT files found in the specified directory.")
    
    # List to hold DataFrames
    df_list = []
    
    for f in files:
        # Read each file
        df = pd.read_csv(f, sep=";")
        # Combine DATE and TIME
        df["time"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
        df = df.set_index("time")
        # Drop unnecessary columns
        df = df.drop(columns=["DATE", "TIME", "TIMEZONE"])
        df_list.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(df_list)
    
    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(combined_df)
    
    # Units and descriptions dictionaries
    units = {
        "P002": "hPa", "TT002": "°C", "TT010": "°C", "TT050": "°C", "TT070": "°C",
        "TT110": "°C", "TT175": "°C", "TT250": "°C", "TT280": "°C",
        "RH002": "%", "RH010": "%", "RH050": "%", "RH110": "%", "RH175": "%",
        "RH250": "%", "RH280": "%",
        "DT002": "°C", "DT010": "°C", "DT050": "°C", "DT110": "°C",
        "DT175": "°C", "DT250": "°C", "DT280": "°C",
        "TS000": "°C", "RR": "mm",
        "FF010": "m/s", "FB010": "m/s", "FU010": "m/s", "FV010": "m/s", "DD010": "°",
        "FF050": "m/s", "FB050": "m/s", "FU050": "m/s", "FV050": "m/s", "DD050": "°",
        "FF110": "m/s", "FB110": "m/s", "FU110": "m/s", "FV110": "m/s", "DD110": "°",
        "FF175": "m/s", "FB175": "m/s", "FU175": "m/s", "FV175": "m/s", "DD175": "°",
        "FF250": "m/s", "FB250": "m/s", "FU250": "m/s", "FV250": "m/s", "DD250": "°",
        "FF280": "m/s", "FB280": "m/s", "FU280": "m/s", "FV280": "m/s", "DD280": "°"
    }
    
    # Assign attributes to each variable
    for var in ds.data_vars:
        if var in units:
            ds[var].attrs["units"] = units[var]
    
    # Save to NetCDF
    #output_file = os.path.join(filepath, outputname)
    #ds.to_netcdf(output_file)
    
    #print(f"NetCDF file saved as: {output_file}")
    return ds
############################################################################################
############################################################################################
# Seasonal mean with weights, modified from https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
def season_mean_yearly(ds):
    """
    Weighted seasonal means for each year, using monthly-length weights.
    Produces one seasonal mean per season per year.
    """

    # Month lengths
    month_length = ds.time.dt.days_in_month

    # Compute weights for each (year, season) combination
    wgts = (
        month_length.groupby(["time.year", "time.season"]) /
        month_length.groupby(["time.year", "time.season"]).sum()
    )

    # Validate weight normalization (sum = 1 for each year+season)
    np.testing.assert_allclose(
        wgts.groupby(["time.year", "time.season"]).sum(), 1.0
    )

    # Mask missing values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Weighted numerator
    ds_sum = (ds * wgts).groupby(["time.year", "time.season"]).sum("time")

    # Weighted denominator
    ones_sum = (ones * wgts).groupby(["time.year", "time.season"]).sum("time")

    return ds_sum / ones_sum
############################################################################################
############################################################################################
def season_mean_by_seasonyear(ds):
    """
    Weighted seasonal means per 'season-year' using month-length weights.
    December is assigned to the *next* year so that DJF stays intact.
    Output dims: (season_year, season, ...)
    """

    # Construct season-year: Dec -> next year, all others -> same year
    season_year = xr.where(
        ds.time.dt.month == 12,
        ds.time.dt.year + 1,
        ds.time.dt.year
    )

    # FIX: assign coordinate using raw data, not a DataArray
    ds = ds.assign_coords(season_year=("time", season_year.data))

    def _season_mean_one_seasonyear(dsy):
        # month-length weights within this season-year
        ml = dsy.time.dt.days_in_month
        w = ml.groupby("time.season") / ml.groupby("time.season").sum()

        # mask-multipliers for NaNs
        ones = xr.where(dsy.isnull(), 0.0, 1.0)

        num = (dsy * w).groupby("time.season").sum("time")
        den = (ones * w).groupby("time.season").sum("time")

        return num / den

    # process per season-year (one key → safe)
    out = ds.groupby("season_year").apply(_season_mean_one_seasonyear)

    # dims: (season_year, season, ...)
    return out
############################################################################################
############################################################################################
# ------------------------------------------------------------
# Metric functions (work with numpy or xarray)
# ------------------------------------------------------------
def rmse(obs, mod):
    return float(np.sqrt(np.mean((mod - obs)**2)))

def mae(obs, mod):
    return float(np.mean(np.abs(mod - obs)))

def bias(obs, mod):
    return float(np.mean(mod - obs))

def correlation(obs, mod):
    return float(np.corrcoef(obs, mod)[0, 1])

def amplitude(x):
    return float(np.max(x) - np.min(x))

def amplitude_error(obs, mod):
    return amplitude(mod) - amplitude(obs)

# ------------------------------------------------------------
# Fully automated diurnal metrics function
# ------------------------------------------------------------
def diurnal_metrics(obs_da, mod_da):
    """
    Compute RMSE, MAE, Bias, Correlation, Amplitude Error
    for 24-hour diurnal mean DataArrays.

    obs_da and mod_da must be 24-element xarray.DataArray or numpy arrays.
    """

    # convert to numpy
    obs = obs_da.values if isinstance(obs_da, xr.DataArray) else np.asarray(obs_da)
    mod = mod_da.values if isinstance(mod_da, xr.DataArray) else np.asarray(mod_da)

    return {
        "RMSE": rmse(obs, mod),
        "MAE": mae(obs, mod),
        "Bias": bias(obs, mod),
        "Correlation": correlation(obs, mod),
        "Amplitude Error": amplitude_error(obs, mod),
    }
# ------------------------------------------------------------
# Raw (hourly) metrics for full time series
# ------------------------------------------------------------
def nse(obs, mod):
    """
    Nash–Sutcliffe Efficiency (NSE).
    1 = perfect
    0 = model = mean
    <0 = worse than using mean
    """
    return float(1 - np.sum((mod - obs)**2) / np.sum((obs - np.mean(obs))**2))
def raw_metrics(obs_da, mod_da):
    """
    Metrics for raw hourly timeseries (no diurnal-specific metrics).
    """
    obs = obs_da.values if isinstance(obs_da, xr.DataArray) else np.asarray(obs_da)
    mod = mod_da.values if isinstance(mod_da, xr.DataArray) else np.asarray(mod_da)
    return {
        "RMSE": rmse(obs, mod),
        "MAE": mae(obs, mod),
        "Bias": bias(obs, mod),
        "Correlation": correlation(obs, mod),
        "NSE": nse(obs, mod),
    }
############################################################################################
############################################################################################
def df_to_latex_categ(
        df,
        caption="Seasonal diurnal-cycle metrics",
        label="tab:metrics"
    ):
    """
    Styling:
    - Bold  = good performance
    - Underline = poor performance
    - Normal = intermediate
    """

    import numpy as np
    df_fmt = df.copy()

    # --- THRESHOLDS --------------------------------------------------------
    # GOOD = bold
    # POOR = underline
    # MID  = normal

    def fmt_rmse(x):
        if x < 0.6: return f"\\textbf{{{x:.2f}}}"
        if x > 1.2: return f"\\underline{{{x:.2f}}}"
        return f"{x:.2f}"

    def fmt_mae(x):
        if x < 0.5: return f"\\textbf{{{x:.2f}}}"
        if x > 1.0: return f"\\underline{{{x:.2f}}}"
        return f"{x:.2f}"

    def fmt_bias(x):
        ax = abs(x)
        if ax < 0.3: return f"\\textbf{{{x:.2f}}}"
        if ax > 0.8: return f"\\underline{{{x:.2f}}}"
        return f"{x:.2f}"

    def fmt_corr(x):
        if x >= 0.85: return f"\\textbf{{{x:.2f}}}"
        if x < 0.6: return f"\\underline{{{x:.2f}}}"
        return f"{x:.2f}"

    def fmt_amp(x):
        ax = abs(x)
        if ax < 0.5: return f"\\textbf{{{x:.2f}}}"
        if ax > 1.0: return f"\\underline{{{x:.2f}}}"
        return f"{x:.2f}"

    # --- APPLY FORMATTING --------------------------------------------------
    df_fmt["RMSE"] = df["RMSE"].apply(fmt_rmse)
    df_fmt["MAE"] = df["MAE"].apply(fmt_mae)
    df_fmt["Bias"] = df["Bias"].apply(fmt_bias)
    df_fmt["Correlation"] = df["Correlation"].apply(fmt_corr)
    df_fmt["Amplitude Error"] = df["Amplitude Error"].apply(fmt_amp)

    # --- GENERATE LATEX ----------------------------------------------------
    latex = df_fmt.to_latex(
        index=False,
        escape=False,   # allow \textbf and \underline
        column_format="l c c r r r r r",
        caption=caption,
        label=label
    )

    # --- MAKE IT TABLE* FOR GMD -------------------------------------------
    latex = latex.replace("\\begin{table}", "\\begin{table*}")
    latex = latex.replace("\\end{table}", "\\end{table*}")

    # --- REPLACE BOOKTABS RULES WITH \hline -------------------------------
    latex = latex.replace("\\toprule", "\\hline")
    latex = latex.replace("\\midrule", "\\hline")
    latex = latex.replace("\\bottomrule", "\\hline")

    return latex
############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################

############################################################################################
############################################################################################