import numpy as np
import matplotlib.pyplot as plt
from csuram import RadarConfig
configdat = RadarConfig.RadarConfig(dz='DBZ')
import pickle
import warnings
warnings.filterwarnings('ignore')
import glob
import os
from pathlib import Path
import matplotlib.colors as colors
import pyart
from datetime import datetime
from scipy.ndimage import label, generate_binary_structure, center_of_mass

import pandas as pd
from CSU_RadarTools import csu_radartools
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain, 
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

import xarray as xr
from copy import deepcopy

from termcolor import colored


from math import radians, cos, sin, asin, sqrt,atan2,degrees

import re


def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd;

def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

def parse_dms(dms):
    parts = re.split('[^\d\w\.]+', dms)
    lat= dms2dd(parts[0], parts[1], parts[2], parts[3])

    return (lat)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    bearing = atan2(sin(lon2-lon1)*cos(lat2), cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1))
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360

    return c * r,bearing


def sloc (lat1,lon1,lat2,lon2):
    re = 6378.0
    c = np.pi/180.0
    lat1 = lat1*c
    lat2 = lat2*c
    lon1 = lon1*c
    lon2 = lon2*c

    d = re*np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2))
    return np.round(d,decimals = 2)

def get_data(scansets_rhi,scannum,nam):
    gnam = f'{nam}_gfile1'
    gnam2 = f'{nam}_gfile2'
    gnam3 = f'{nam}_gfile3'
    az = f'{nam}_az'
    tm= f'{nam}_time'
    #print(scansets_rhi[scannum].keys())
    cf = scansets_rhi[scannum][nam]
    rhi={'cf':cf,
            'gfile1':scansets_rhi[scannum][gnam],
            'gfile2':scansets_rhi[scannum][gnam2],
            'gfile3':scansets_rhi[scannum][gnam3],
            'time':scansets_rhi[scannum][tm],
            'az':scansets_rhi[scannum][az]
        }
    return rhi


def get_sdat(dat,az=0.0):
    x = dat['x0'].values
    y = dat['y0'].values
    z = dat['z0'].values
    dz = np.squeeze(dat['filtered_refectivity'].values)
    whgd =  np.where(dz>-10)
    dumzr = np.squeeze(dat['differential_reflectivity'].values)
    zr = np.zeros_like(dumzr)*np.nan
    zr[whgd] = dumzr[whgd]
    dumrho = np.squeeze(dat['cross_correlation_ratio'].values)
    dumkdp = np.squeeze(dat['specific_differential_phase'].values)
    dumsw = np.squeeze(dat['spectrum_width'].values)
    rho = np.zeros_like(dumrho)*np.nan
    rho[whgd]=dumrho[whgd]
    kdp = np.zeros_like(dumkdp)*np.nan
    kdp[whgd]=dumkdp[whgd]
    sw= np.zeros_like(dumsw)*np.nan
    sw[whgd]=dumsw[whgd]
    
    ve = np.squeeze(dat['CV'].values)
    lat0 = dat['lat0'].values
    lon0 = dat['lon0'].values
    tm = dat['time'].values[0]
    dattim= pd.to_datetime(str(tm)).replace(tzinfo=None)
    #dattim = pd.to_datetime(str(tm))
    


    data = {'x':x,'y':y,'z':z,'dz':dz,'ve':ve,'lat':lat0,'lon':lon0,'time':dattim,'az':az,
    'rho':rho,'kdp':kdp,'sw':sw,'zdr':zr}#
    return data

def get_data_rhi(scansets_rhi,scannum,nam):
    gnam = f'{nam}_gfile1'
    gnam2 = f'{nam}_gfile2'
    gnam3 = f'{nam}_gfile3'
    az = f'{nam}_az'
    tm= f'{nam}_time'
    #print(scansets_rhi[scannum].keys())
    cf = scansets_rhi[scannum][nam]
    rhi={'cf':cf,
         'gfile1':scansets_rhi[scannum][gnam],
         'gfile2':scansets_rhi[scannum][gnam2],
         'gfile3':scansets_rhi[scannum][gnam3],
         'time':scansets_rhi[scannum][tm],
         'az':scansets_rhi[scannum][az]
        }
    return rhi



def get_data(scansets_rhi,scannum,nam):
    gnam = f'{nam}_gfile1'
    gnam2 = f'{nam}_gfile2'
    gnam3 = f'{nam}_gfile3'
    az = f'{nam}_az'
    tm= f'{nam}_time'
    #print(scansets_rhi[scannum].keys())
    cf = scansets_rhi[scannum][nam]
    rhi={'cf':cf,
         'gfile1':scansets_rhi[scannum][gnam],
         'gfile2':scansets_rhi[scannum][gnam2],
         'gfile3':scansets_rhi[scannum][gnam3],
         'time':scansets_rhi[scannum][tm],
         'az':scansets_rhi[scannum][az]
        }
    return rhi


def print_sets_three(scansets,start_val,full=True):
    ###5th time -- not good
    normcol='green'
    miscol='blue'
    badcol = 'red'
    col=normcol

    chivo_scanset = str(f'{start_val:0>3}')

    chivo_0_rhi ='rhi1'
    
    try:
        chivo_0 = get_data_rhi(scansets,chivo_scanset,chivo_0_rhi)
    except:
        col=badcol
        print(colored('missing',badcol))
        return

    chivo_scanset_20 = str(f'{start_val+1:0>3}')
    chivo_20_rhi ='rhi1'
    try:
        chivo_20 = get_data_rhi(scansets,chivo_scanset_20,chivo_20_rhi)
        delT20=(chivo_20['time']-chivo_0['time']).seconds
        delaz20=chivo_20['az']-chivo_0['az']
        if np.abs(delaz20)>0.1:
            col=miscol
    except KeyError as ke:
        col=miscol
        delT20 = '9:99:99'
        delaz20 = -9.9

    #print('delT20:',delT20,'delaz20:',delaz20)

    chivo_scanset_60 = str(f'{start_val+2:0>3}')
    chivo_60_rhi ='rhi1'
    try:
        chivo_60 = get_data_rhi(scansets,chivo_scanset_60,chivo_60_rhi)

        delT60=(chivo_60['time']-chivo_0['time']).seconds
        delaz60=chivo_60['az']-chivo_0['az']
        if np.abs(delaz60)>0.1:
            col=miscol
    except KeyError as ke:
        col=miscol
        delT60 = '9:99:99'
        delaz60 = -9.9
    #print('delT60:',delT60,'delaz20:',delaz60)

    chivo_scanset_90 = str(f'{start_val+3:0>3}')
    chivo_90_rhi ='rhi1'
    try:
        chivo_90 = get_data_rhi(scansets,chivo_scanset_90,chivo_90_rhi)
        delT90=(chivo_90['time']-chivo_0['time']).seconds
        delaz90=chivo_90['az']-chivo_0['az']
        if np.abs(delaz90)>0.1:
            col=miscol
    except KeyError as ke:
        col=miscol
        delT90 = '9:99:99'
        delaz90 = -9.9
    #print('delT90:',delT90,'delaz90:',delaz90)
    t1=chivo_0['time']
    
   # print('full values is ',full)
    if full == True:
        print(colored(f'{chivo_scanset} {t1:%H%M%S}',col),'delT20:',delT20,'delaz20:',delaz20,
                    'delT60:',delT60,'delaz60:',delaz60,
                    'delT90:',delT90,'delaz90:',delaz90)

    else:
        if col is normcol:
            print(colored(f'{chivo_scanset} {t1:%H%M%S}',col),'delT20:',delT20,'delaz20:',delaz20,
                        'delT60:',delT60,'delaz60:',delaz60,
                        'delT90:',delT90,'delaz90:',delaz90)
       
