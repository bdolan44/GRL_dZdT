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


from CSU_RadarTools import csu_radartools
from csu_radartools import (csu_fhc, csu_liquid_ice_mass, csu_blended_rain, 
                            csu_dsd, csu_kdp, csu_misc, fundamentals)

import xarray as xr
from copy import deepcopy

from termcolor import colored

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




def check_sounding_for_montonic(sounding):
    """
    So the sounding interpolation doesn't fail, force the sounding to behave
    monotonically so that z always increases. This eliminates data from
    descending balloons.
    """
    snd_T = sounding['temp']  # In old SkewT, was sounding.data
    snd_z = sounding['hght']  # In old SkewT, was sounding.data
    dummy_z = []
    dummy_T = []
    #print(snd_T.mask)
    if not snd_T.mask.all(): #May cause issue for specific soundings
        dummy_z.append(snd_z[0])
        dummy_T.append(snd_T[0])
        for i, height in enumerate(snd_z):
            if i > 0:
                if snd_z[i] > snd_z[i-1]:# and not snd_T.mask[i]:
                    if np.isfinite(snd_z[i]) and np.isfinite(snd_T[i]):
                        #print(snd_T[i])
                        dummy_z.append(snd_z[i])
                        dummy_T.append(snd_T[i])
        snd_z = np.array(dummy_z)
        snd_T = np.array(dummy_T)
    else:
        print('uh-oh. sounding problem')
    return snd_T, snd_z

def interpolate_sounding_to_grid(sounding, radar_z, dz):
    """Takes sounding data and interpolates it to every radar gate."""
    
    #radar_z = get_z_from_radar(radar)
    radar_T = None
    press = sounding['pres']
    snd_T, snd_z = check_sounding_for_montonic(sounding)
    #print (snd_z)
    whgd = np.where(np.logical_and(np.array(snd_T)>-99,np.array(snd_z)>0.))
    shape = np.shape(dz)
#    print 'Shape: ',shape
    rad_z1d = radar_z*1000.
    rad_T1c = np.interp(rad_z1d, snd_z[whgd], snd_T[whgd])
    rad_pc = np.interp(rad_z1d,snd_z[whgd],press[whgd])
    #print(snd_z,snd_T)
    tfixed=np.zeros(dz.shape)
    zfixed=np.zeros(dz.shape)
    pfixed=np.zeros(dz.shape)
    for i in range(len(radar_z)):
        tfixed[i,:] = rad_T1c[i]
        zfixed[i,:] = rad_z1d[i]
        pfixed[i,:] = rad_pc[i]

    return tfixed, zfixed, pfixed


def get_microphys(dat,sndT,sndZ):

    dz1=dat['dz']
    dr1 =dat['zdr']
    kd1=dat['kdp']
    rh1=dat['rho']
    rng = dat['y']
    radz = dat['z']
    whbad = np.where(np.isnan(dz1))

    fh = csu_fhc.csu_fhc_summer(dz=dz1, zdr=dr1, rho=rh1, kdp=kd1, use_temp=True, band='C',
                                    T=sndT,use_trap=True)
    fh[whbad]=-1

    mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(dz1, dr1, sndZ, T=sndT)
    dat['mw']=mw
    dat['mi']=mi
    dat['fh']=fh
    dat['sndT']=sndT
    dat['sndZ'] = sndZ
    return dat


def plot_diffs_w(rhi1,rhi2,rhi3,rhi4,diff1,diff2,diff3,res='100m',basedir='./',xlim=[0,100],ylim=[0,15],extra='',radnam='chivo'):
    fig, ax = plt.subplots(3,4,figsize=(20,15))
    axf =ax.flatten()

    f=0
    c = axf[f].pcolormesh(rhi1['y'],rhi1['z'],rhi1['dz'],vmin=0,vmax=80,cmap=configdat.temp_cmap)
    plt.colorbar(c,ax=axf[f])
    CS = axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    axf[f].set_title("delT = 0",fontsize=25)

    axf[f].clabel(CS, CS.levels, inline=True, fontsize=10)

    f=1
    c= axf[f].pcolormesh(rhi2['y'],rhi2['z'],rhi2['dz'],vmin=0,vmax=80,cmap=configdat.temp_cmap)
    plt.colorbar(c,ax=axf[f])

    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    delT1 =diff1['delT']
    axf[f].set_title(f"delT = {delT1}",fontsize=25)

    f=2
    c= axf[f].pcolormesh(rhi3['y'],rhi3['z'],rhi3['dz'],vmin=0,vmax=80,cmap=configdat.temp_cmap)
    plt.colorbar(c,ax=axf[f])
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    delT3 =diff2['delT']
    axf[f].set_title(f"delT = {delT3}",fontsize=25)

    f=3
    c= axf[f].pcolormesh(rhi4['y'],rhi4['z'],rhi4['dz'],vmin=0,vmax=80,cmap=configdat.temp_cmap)
    plt.colorbar(c,ax=axf[f])

    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    delT4 =diff3['delT']
    axf[f].set_title(f"delT = {delT4}",fontsize=25)
    f=4
    axf[f].axis('off')

    f=5
    c=axf[f].pcolormesh(diff1['rhi1y'],diff1['rhi1hgt'],diff1['dzdiff'],vmin=-20,vmax=20,cmap='bwr')
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    plt.colorbar(c,ax=axf[f])

    #axf[f].contour(rhi0slant1['rhi1y'],rhi0slant1['rhi1hgt'],rhi0slant1['dz'],levels=[5,35,60],colors=['k'],linestyles=['-'])
    axf[f].set_title(f"delT = {delT1}",fontsize=25)

    f=6
    c=axf[f].pcolormesh(diff2['rhi1y'],diff2['rhi1hgt'],diff2['dzdiff'],vmin=-20,vmax=20,cmap='bwr')
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    #axf[f].contour(rhi0slant1['rhi1y'],rhi0slant1['rhi1hgt'],rhi0slant1['dz'],levels=[5,35,60],colors=['k'],linestyles=['-'])
    axf[f].set_title(f"delT = {delT3}",fontsize=25)
    plt.colorbar(c,ax=axf[f])

    f=7
    c=axf[f].pcolormesh(diff3['rhi1y'],diff3['rhi1hgt'],diff3['dzdiff'],vmin=-20,vmax=20,cmap='bwr')
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    #axf[f].contour(rhi0slant1['rhi1y'],rhi0slant1['rhi1hgt'],rhi0slant1['dz'],levels=[5,35,60],colors=['k'],linestyles=['-'])
    axf[f].set_title(f"delT = {delT4}",fontsize=25)
    plt.colorbar(c,ax=axf[f])


    for a in axf:
        a.set_xlim(xlim[0],xlim[1])
        a.set_ylim(ylim[0],ylim[1])
        a.set_ylabel('Height (km)',fontsize=15)
        a.set_xlabel('Distance (km)',fontsize=15)
        a.grid()

        
    f=8
    axf[f].axis('off')

    f=9
    c=axf[f].pcolormesh(diff1['rhi1y'],diff1['rhi1hgt'],diff1['w'],vmin=-20,vmax=20,cmap='PuOr_r')
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    axf[f].set_title(f"delT = {delT1}",fontsize=25)
    plt.colorbar(c,ax=axf[f])
    f=10
    c=axf[f].pcolormesh(diff2['rhi1y'],diff2['rhi1hgt'],diff2['w'],vmin=-20,vmax=20,cmap='PuOr_r')
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])
    axf[f].set_title(f"delT = {delT3}",fontsize=25)
    plt.colorbar(c,ax=axf[f])
    f=11
    c=axf[f].pcolormesh(diff3['rhi1y'],diff3['rhi1hgt'],diff3['w'],vmin=-20,vmax=20,cmap='PuOr_r')
    plt.colorbar(c,ax=axf[f])
    axf[f].set_title(f"delT = {delT4}",fontsize=25)
    axf[f].contour(rhi1['y'],rhi1['z'],rhi1['dz'],levels=[5,35,55],colors=['k'],linestyles=['-'])


    tm = rhi1['time']
    plt.suptitle(f'{radnam} {tm:%Y%m%d %H%M%S} {extra}',fontsize=25,y=1.01)
    plt.tight_layout()
    plt.savefig(f'{basedir}{radnam}_{tm:%Y%m%d_%H%M%S}_10panel_{res}_wAllETH_{extra}.png',dpi=500,bbox_inches='tight',facecolor='white')

def plot_microphys(dat,figdir='./',radar='chivo',rhinam = 'rhi1',xmin=0,xmax=100,ymax=15,extra='',radnam='chivo',res='100m'):

    fig, ax = plt.subplots(1,3,figsize=(15,5))
    axf=ax.flatten()

    c= axf[0].pcolormesh(dat['y'],dat['z'],dat['mw'],norm=colors.LogNorm(vmin=0.01, vmax=10),cmap='YlOrRd')
    cb1= plt.colorbar(c,ax=axf[0])
    cb1.set_label('g m$^3$')
    axf[0].set_title("Liquid Water Mass")


    c= axf[1].pcolormesh(dat['y'],dat['z'],dat['mi'],norm=colors.LogNorm(vmin=0.01, vmax=10),cmap='YlOrRd')
    cb2 = plt.colorbar(c,ax=axf[1])
    cb2.set_label('g m$^3$')
    axf[1].set_title("Ice Water Mass")

    c= axf[2].pcolormesh(dat['y'],dat['z'],dat['fh'],vmin=0,vmax=10,cmap=configdat.hid_cmap)
    cb =plt.colorbar(c,ax=axf[2])
    cb.set_ticks(np.arange(1.4, 10, 0.9))
    cb.ax.set_yticklabels(['DZ', 'RN', 'IC', 'AG',
                            'WS', 'VI', 'LDG',
                            'HDG', 'Ha', 'BD'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    axf[2].set_title('HID')

    for a in axf:
        a.set_xlim(xmin,xmax)
        a.set_ylabel('Height (km)')
        a.set_xlabel('Distance (km)')
        
        a.set_ylim(0,ymax)
        a.grid()
    rtime1 = dat['time']
    
    plt.suptitle(f'{rhinam} {rtime1:%Y%m%d %H%M%S} {extra}')
    plt.tight_layout()
    plt.savefig(f'{figdir}{radar}{rtime1:%Y%m%d_%H%M%S}_{rhinam}_microphsyics_{res}_{extra}_{radnam}.png',dpi=500,facecolor='white',bbox_inches='tight')

def plot_polvar(dat,figdir='./',radar='chivo',rhinam = 'rhi1',xmin=0,xmax=100,ymax=15,res='100m',extra='',radnam='chivo'):
    fig, ax = plt.subplots(3,2, figsize=(15,15))
    axf = ax.flatten()

    c=axf[0].pcolormesh(dat['y'],dat['z'],dat['dz'],vmin=0,vmax=80,cmap=configdat.temp_cmap)
    plt.colorbar(c,ax=axf[0])
    axf[0].set_title('dz')


    c=axf[1].pcolormesh(dat['y'],dat['z'],dat['zdr'],vmin=-1,vmax=5,cmap=configdat.zdr_cmap)
    plt.colorbar(c,ax=axf[1])
    axf[1].set_title('zdr')

    c=axf[2].pcolormesh(dat['y'],dat['z'],dat['kdp'],vmin=-1,vmax=5,cmap=configdat.kdp_cmap)
    plt.colorbar(c,ax=axf[2])
    axf[2].set_title('kdp')


    c=axf[3].pcolormesh(dat['y'],dat['z'],dat['rho'],vmin=0.8,vmax=1,cmap='Reds')
    plt.colorbar(c,ax=axf[3])
    axf[3].set_title('Rho')


    c=axf[4].pcolormesh(dat['y'],dat['z'],dat['ve'],vmin=-20,vmax=20,cmap='pyart_NWSVel')
    plt.colorbar(c,ax=axf[4])
    axf[4].set_title('VR')


    c=axf[5].pcolormesh(dat['y'],dat['z'],dat['sw'],vmin=0,vmax=5,cmap='gnuplot')
    plt.colorbar(c,ax=axf[5])
    axf[5].set_title('SW')

    for a in axf:
        a.set_xlim(xmin,xmax)
        a.set_ylim(0,ymax)
        a.grid()
        a.set_ylabel('Height (km)')
        a.set_xlabel('Distance (km)')
    
    rtime1 = dat['time']
    
    plt.suptitle(f'{rhinam} {rtime1:%Y%m%d %H%M%S} {extra}')
    plt.tight_layout()
    plt.savefig(f'{figdir}{radar}{rtime1:%Y%m%d_%H%M%S}_{rhinam}_poldat_{res}_{extra}_{radnam}.png',dpi=500,facecolor='white',bbox_inches='tight')


def get_slant_ind(x,y,ang,rng):
    thetan = np.deg2rad(ang)
    yslant = np.cos(np.deg2rad(ang))*rng
    xslant = np.sin(np.deg2rad(ang))*rng

    xvals = []
    for i,v in enumerate(xslant):
        whx = np.squeeze(np.where(np.isclose(v,x,atol=0.05)))
        #print(np.shape(whx))
        if np.ndim(whx) == 0:
            xvals.append(whx)

        else:
            try: 
                xvals.append(whx[0])

            except IndexError as ie:
                xvals.append(-999)

    xvals = np.array(xvals)   

    yvals = []
    for i,v in enumerate(yslant):
        why = np.squeeze(np.where(np.isclose(v,y,atol=0.05)))

        #print(why)
        if np.ndim(why) == 0:
            yvals.append(why)

        else:
            try: 
                yvals.append(why[0])

            except IndexError as ie:
                yvals.append(-999)
    yvals = np.array(yvals)   

    return xvals,yvals


def print_sets(scansets,start_val):
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

    chivo_scanset_20 = str(f'{start_val:0>3}')
    chivo_20_rhi ='rhi3'
    try:
        chivo_20 = get_data_rhi(scansets,chivo_scanset_20,chivo_20_rhi)
        delT20=(chivo_20['time']-chivo_0['time']).seconds
        delaz20=chivo_20['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT20 = '9:99:99'
        delaz20 = -9.9

    #print('delT20:',delT20,'delaz20:',delaz20)

    chivo_scanset_60 = str(f'{start_val+1:0>3}')
    chivo_60_rhi ='rhi1'
    try:
        chivo_60 = get_data_rhi(scansets,chivo_scanset_60,chivo_60_rhi)

        delT60=(chivo_60['time']-chivo_0['time']).seconds
        delaz60=chivo_60['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT60 = '9:99:99'
        delaz60 = -9.9
    #print('delT60:',delT60,'delaz20:',delaz60)

    chivo_scanset_90 = str(f'{start_val+1:0>3}')
    chivo_90_rhi ='rhi3'
    try:
        chivo_90 = get_data_rhi(scansets,chivo_scanset_90,chivo_90_rhi)
        delT90=(chivo_90['time']-chivo_0['time']).seconds
        delaz90=chivo_90['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT90 = '9:99:99'
        delaz90 = -9.9
    #print('delT90:',delT90,'delaz90:',delaz90)


    chivo_scanset_120 = str(f'{start_val+2:0>3}')
    chivo_120_rhi ='rhi1'
    try:
        chivo_120 = get_data_rhi(scansets,chivo_scanset_120,chivo_120_rhi)
        delT120=(chivo_120['time']-chivo_0['time']).seconds
        delaz120=chivo_120['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT120 = '9:99:99'
        delaz120 = -9.9
    #print('delT120:',delT120,'delaz120:',delaz120)


    chivo_scanset_150 = str(f'{start_val+2:0>3}')
    chivo_150_rhi ='rhi3'
    try:
        chivo_150 = get_data_rhi(scansets,chivo_scanset_150,chivo_150_rhi)
        delT150=(chivo_150['time']-chivo_0['time']).seconds
        delaz150=chivo_150['az']-chivo_0['az']
    except:
        col=miscol
        delT150='9:99:99'
        delaz150=-9.9
    
    print(colored(chivo_scanset,col),'delT20:',delT20,'delaz20:',delaz20,
                'delT60:',delT60,'delaz60:',delaz60,
                'delT90:',delT90,'delaz90:',delaz90,
                'delT120:',delT120,'delaz120:',delaz120,
                'delT150:',delT150,'delaz150:',delaz150)


def print_sets_two(scansets,start_val):
    ###5th time -- not good
    normcol='green'
    miscol='blue'
    badcol = 'red'
    col=normcol

    chivo_scanset = str(f'{start_val:0>3}')

    chivo_0_rhi ='rhi3'
    
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
    except KeyError as ke:
        col=miscol
        delT20 = '9:99:99'
        delaz20 = -9.9

    #print('delT20:',delT20,'delaz20:',delaz20)

    chivo_scanset_60 = str(f'{start_val+1:0>3}')
    chivo_60_rhi ='rhi3'
    try:
        chivo_60 = get_data_rhi(scansets,chivo_scanset_60,chivo_60_rhi)

        delT60=(chivo_60['time']-chivo_0['time']).seconds
        delaz60=chivo_60['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT60 = '9:99:99'
        delaz60 = -9.9
    #print('delT60:',delT60,'delaz20:',delaz60)

    chivo_scanset_90 = str(f'{start_val+2:0>3}')
    chivo_90_rhi ='rhi1'
    try:
        chivo_90 = get_data_rhi(scansets,chivo_scanset_90,chivo_90_rhi)
        delT90=(chivo_90['time']-chivo_0['time']).seconds
        delaz90=chivo_90['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT90 = '9:99:99'
        delaz90 = -9.9
    #print('delT90:',delT90,'delaz90:',delaz90)


    chivo_scanset_120 = str(f'{start_val+2:0>3}')
    chivo_120_rhi ='rhi3'
    try:
        chivo_120 = get_data_rhi(scansets,chivo_scanset_120,chivo_120_rhi)
        delT120=(chivo_120['time']-chivo_0['time']).seconds
        delaz120=chivo_120['az']-chivo_0['az']
    except KeyError as ke:
        col=miscol
        delT120 = '9:99:99'
        delaz120 = -9.9
    #print('delT120:',delT120,'delaz120:',delaz120)


    chivo_scanset_150 = str(f'{start_val+3:0>3}')
    chivo_150_rhi ='rhi3'
    try:
        chivo_150 = get_data_rhi(scansets,chivo_scanset_150,chivo_150_rhi)
        delT150=(chivo_150['time']-chivo_0['time']).seconds
        delaz150=chivo_150['az']-chivo_0['az']
    except:
        col=miscol
        delT150='9:99:99'
        delaz150=-9.9
    
    print(colored(chivo_scanset,col),'delT20:',delT20,'delaz20:',delaz20,
                'delT60:',delT60,'delaz60:',delaz60,
                'delT90:',delT90,'delaz90:',delaz90,
                'delT120:',delT120,'delaz120:',delaz120,
                'delT150:',delT150,'delaz150:',delaz150)

def print_sets_three(scansets,start_val):
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
    
    print(colored(f'{chivo_scanset} {t1:%H%M%S}',col),'delT20:',delT20,'delaz20:',delaz20,
                'delT60:',delT60,'delaz60:',delaz60,
                'delT90:',delT90,'delaz90:',delaz90)




def get_slant_data(dz,zvals,yvals,xvals):
    if np.ndim(dz) != 3:
        print('Be sure to enter a 3D dz array (z,y,x)')
    dznew = np.zeros([len(zvals),len(yvals)])*np.nan
    #print(np.shape(dzsub))
    for i,v in enumerate(yvals):
        #print(i)
        #print(np.array(yvals)[i])
        if yvals[i] ==-999 or xvals[i] == -999:
            pass
        else:
            dznew[:,i]=dz[:,yvals[i],xvals[i]]
        
    return dznew
    
def get_dz_data(data,dzname):
    dznew = data[dzname][0,:,:,-1].values
    return dznew
    
def get_data(scanset,scannum,rhinum,cfdir,griddir1,griddir3):
    whscan1 = np.where(np.char.equal(scanset[rhinum]['scannum'], scannum))[0][0]
    print(whscan1)
    rhi1f = scanset[rhinum]['cfnames'][whscan1]
    rhi1 = f'{cfdir}{rhi1f}'
    rhi1az = scanset[rhinum]['az1'][whscan1]
    gfiled3 = scanset[rhinum]['gridnames3'][whscan1]
    gfile3 = f'{griddir3}{gfiled3}'

    gfiled1 = scanset[rhinum]['gridnames1'][whscan1]
    gfile1 = f'{griddir1}{gfiled1}'

    pfile1 = scanset['ppi']['cfnames'][whscan1]
    pfile = f'{cfdir}{pfile1}'
    r1time = scanset[rhinum]['times'][whscan1]
    ptime = scanset['ppi']['times'][whscan1]
    pfileg = scanset['ppi']['gridnames1'][whscan1]
    data = {'cf':rhi1,'az':rhi1az,'grid1':gfiled1,'grid3':gfiled3,'time':r1time,'ptime':ptime,'pfile':pfile,'scanset':scannum}
    return data
    
def get_all_eth(dat):
    eth = []
    for i,v in enumerate(np.arange(-10,75,0.5)):
        eth.append(get_height(dat['dz'],dat['z'],v))
    dat['all_eth']=np.array(eth)

    return dat

def get_w_eth2d(dat1,dat2,diffdat):
    delT= (dat2['time']-dat1['time']).seconds
    ethdiff = (np.array(dat2['all_eth'])-np.array(dat1['all_eth']))*1000./delT
    dzdiff =np.array(dat1['dz'])-np.array(dat1['dz'])
    w = np.zeros_like(dat1['dz'])*np.nan
    diffdat['all_ethdiff']=ethdiff
    for i,z in enumerate(dat1['z']):
        for j, t in enumerate(dat1['y']):
            whi = np.where(np.array(dat2['all_eth'])[:,j] == z)[0]
            if len(whi)>0:
                w[i,j] = np.array(ethdiff[:,j])[whi[-1]]
    diffdat['w']=w
    return diffdat



def get_diffs_lin(dz1,dz2):
    #we need special handling for the regions where one is nan and the other is not
    whdz1 = np.where(np.logical_and(np.isfinite(dz1),np.isnan(dz2)))    
    whdz2 = np.where(np.logical_and(np.isfinite(dz2),np.isnan(dz1)))
 
    #and since log reflectivity can be negative, we need to take the absolute value in these areas for the change.
    dzlin1 = np.zeros_like(dz1)
    wh1 =np.where(np.isfinite(dz1))
    dzlin1[wh1]=10.**(dz1[wh1]/10.)
    
    dzlin2 = np.zeros_like(dz2)
    wh2 =np.where(np.isfinite(dz2))
    dzlin2[wh2]=10.**(dz2[wh2]/10.)

    diff_dz_lin = dzlin2-dzlin1
    print(np.nanmax(diff_dz_lin),np.nanmin(diff_dz_lin))
    whpos = np.where(diff_dz_lin>0)
    whneg = np.where(diff_dz_lin<0)
    print(diff_dz_lin[whneg])
    whzero = np.where(diff_dz_lin==0)
    
    diff_dzd=np.zeros_like(diff_dz_lin)*np.nan
    diff_dzd[whpos]=np.log10(diff_dz_lin[whpos])
    diff_dzd[whneg]=-1*np.log10(np.abs(diff_dz_lin[whneg]))
    
    print(np.nanmin(diff_dzd),np.nanmax(diff_dzd))
    return diff_dzd

def match_times(ctimes,gtimes,cfiles,gfiles3,gfiles1,scanset,rtype):
    gmatch = []
    gmatch1 = []
    for i,c in enumerate(ctimes):
        diff =abs(gtimes - c)
        whd = np.where(diff == np.min(diff))[0][0]
        #print(gfiles[whd])
        dum = Path(gfiles3[whd]).name
        dum1 = Path(gfiles1[whd]).name
        #print(dum)
        gmatch.append(dum)
        gmatch1.append(dum1)
        #print(cfiles[i],'\n',gfiles[whd])
    scanset[rtype]['gridnames3']=np.array(gmatch)
    scanset[rtype]['gridnames1']=np.array(gmatch1)


def get_height(dz,zval,thresh):
    
    hgt = np.zeros_like(dz)
    for i,v in enumerate(dz[0,:]):
        hgt[:,i] = zval
    
    wheth = np.where(dz>thresh)
    etcarr = np.zeros_like(hgt)*np.nan
    etcarr[wheth]=hgt[wheth]
    eth = np.nanmax(etcarr,axis=0)

    return eth
    
def get_az_diff(scanset):
    rhi1 = np.array(scanset['rhi1']['az1'])
    rhi2 = np.array(scanset['rhi2']['az1'])
    rhi3 = np.array(scanset['rhi3']['az1'])
    rhi4 = np.array(scanset['rhi4']['az1'])


    diff12 = rhi1-rhi2
    diff13 = rhi1-rhi3
    diff14 = rhi1-rhi4
    
    print(diff12)
    print(diff13)
    print(diff14)
    
def change_case(rfiles):
    rfiles_new = []
    for i,r in enumerate(rfiles):
        strpl =rfiles[i].split('rhi')
        newstr = f'{strpl[0]}RHI{strpl[1]}'
        rfiles_new.append(newstr)
    return np.array(rfiles_new)
    
def plot_raw_ppi(rhi_0,rhi_20, rhi_90,rhi_120,scannum,xlim=[30,60],ylim=[0,15],basedir='./',extra=''):
    fig = plt.figure(figsize=(10, 20))

    rng = np.arange(0,100,1)
    
    rhi0 = pyart.io.read(rhi_0['pfile'])
    rhi20 = pyart.io.read(rhi_20['pfile'])
    rhi90 = pyart.io.read(rhi_90['pfile'])
    rhi120 = pyart.io.read(rhi_120['pfile'])
    colorbar_label = '(dBZ)'

    display = pyart.graph.RadarDisplay(rhi0)
#    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    display.plot_ppi('filtered_refectivity', 1, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax1.set_ylabel('hgt',fontsize=18)
    az1=rhi_0['az']
    T0 =rhi_0['ptime']
    delT0 = 0
    x1 = np.sin(np.deg2rad(az1))*rng
    y1 = np.cos(np.deg2rad(az1))*rng

    ax1.plot(x1,y1)
    
    ax1.set_title(f'PPI0 T0:{T0:%H%M%S} delT:{delT0} AZ:{az1:.1f}',fontsize=22)

    ax2 = fig.add_subplot(412)
    display = pyart.graph.RadarDisplay(rhi20)
    display.plot_ppi('filtered_refectivity', 1, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)

    ax2.set_ylabel('hgt',fontsize=18)
    az2=rhi_20['az']
    T1 =rhi_20['ptime']
    delT01 = (T1-T0).seconds
    ax2.set_title(f'PPI2 T1:{T1:%H%M%S} delT:{delT01} AZ:{az2:.1f}',fontsize=22)


    x2 = np.sin(np.deg2rad(az2))*rng
    y2 = np.cos(np.deg2rad(az2))*rng

    ax2.plot(x2,y2)


    ax3 = fig.add_subplot(413)
    display = pyart.graph.RadarDisplay(rhi90)
    display.plot_ppi('filtered_refectivity', 1, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)

    ax3.set_ylabel('hgt',fontsize=18)

    az3=rhi_90['az']
    T2 =rhi_90['ptime']
    delT02 = (T2-T0).seconds
    ax3.set_title(f'PPI3 T2:{T2:%H%M%S} delT:{delT02} AZ:{az3:.1f}',fontsize=22)

    x3 = np.sin(np.deg2rad(az3))*rng
    y3 = np.cos(np.deg2rad(az3))*rng

    ax3.plot(x3,y3)


    ax4 = fig.add_subplot(414)
    display = pyart.graph.RadarDisplay(rhi120)
    display.plot_ppi('filtered_refectivity', 1, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)

    ax4.set_ylabel('hgt',fontsize=18)
    az4=rhi_120['az']
    T3 =rhi_120['ptime']
    delT03 = (T3-T0).seconds
    ax4.set_title(f'PPI4 T3:{T3:%H%M%S} delT:{delT03} AZ:{az4:.1f}',fontsize=22)

    x4 = np.sin(np.deg2rad(az4))*rng
    y4 = np.cos(np.deg2rad(az4))*rng

    ax4.plot(x4,y4)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    plt.suptitle(f'Date {T0:%Y%m%d}',fontsize=22,y=1.01)

    plt.tight_layout()

    plt.savefig(f"{basedir}CSAPR_RAWPPI_4panel_{T0:%Y%m%d_%H%M%S}_{scannum}_{extra}.png",bbox_inches='tight',facecolor='white',dpi=400)    
    
def get_grid_slant(scanset,sctype,scannum,griddir1,griddir3):

    whscan1 = np.where(np.char.equal(scanset[sctype]['scannum'], scannum))[0][0]
    print(whscan1)
    rhi1az = scanset[sctype]['az1'][whscan1]
    gfiled1 = scanset[sctype]['gridnames1'][whscan1]
    gfile1 = f'{griddir1}{gfiled1}'

    data1 = xr.open_dataset(gfile1)


    #rng = np.arange(0,100.1,0.1)
    #dzrhi1gdat = np.squeeze(data1['filtered_refectivity'].values)
    #wrhi1gdat = np.squeeze(data1['W'].values)

    #[tm,:,:,:]
    zrhi1g = data1['z0'].values#/1000.
    xrhi1g = data1['x0'].values#/1000.
    yrhi1g = data1['y0'].values#/1000.

    
#    xvalsrhi1, yvalsrhi1 = get_slant_ind(xrhi1g,yrhi1g,rhi1az, rng)

    dzslantrhi1 = get_dz_data(data1,'filtered_refectivity')
    veslantrhi1 = get_dz_data(data1,'filtered_CV')
    #wslantrhi1 = get_slant_data(wrhi1gdat,zrhi1g,yvalsrhi1,xvalsrhi1)
    # xvalsrhi2, yvalsrhi2 = get_slant_ind(x,y,rhi2az, rng)
    # dzslantrhi2 = get_slant_data(dzsub,zvals,yvalsrhi2,xvalsrhi2)

    rtimed1 = data1.time[0].values
    ts1 = (rtimed1- np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    rtime1 = datetime.utcfromtimestamp(ts1)

    slantdata1 = {'scanset':scanset,
                 'dz':dzslantrhi1,'ve':veslantrhi1,'az':rhi1az,'z':zrhi1g,'x':xrhi1g,'y':yrhi1g,'time':rtime1}
    
    gfiled3 = scanset[sctype]['gridnames3'][whscan1]
    gfile3 = f'{griddir3}{gfiled3}'

    data3= xr.open_dataset(gfile3)


    #rng = np.arange(0,100.1,0.1)
    #dzrhi1gdat = np.squeeze(data1['filtered_refectivity'].values)
    #wrhi1gdat = np.squeeze(data1['W'].values)

    #[tm,:,:,:]
    zrhi3g = data3['z0'].values#/1000.
    xrhi3g = data3['x0'].values#/1000.
    yrhi3g = data3['y0'].values#/1000.

    
#    xvalsrhi1, yvalsrhi1 = get_slant_ind(xrhi1g,yrhi1g,rhi1az, rng)

    dzslantrhi3 = get_dz_data(data3,'filtered_refectivity')
    veslantrhi3 = get_dz_data(data3,'filtered_CV')
    #wslantrhi1 = get_slant_data(wrhi1gdat,zrhi1g,yvalsrhi1,xvalsrhi1)
    # xvalsrhi2, yvalsrhi2 = get_slant_ind(x,y,rhi2az, rng)

    slantdata3 = {'scanset':scanset,
                 'dz':dzslantrhi3,'ve':veslantrhi3,'az':rhi1az,'z':zrhi3g,'x':xrhi3g,'y':yrhi3g,'time':rtime1}
    
    return data1, slantdata1, slantdata3
    
    
def plot_raw(rhi_0,rhi_20, rhi_90,rhi_120,scannum,xlim=[0,30],ylim=[0,15],basedir='./',extra=''):
    fig = plt.figure(figsize=(10, 20))

    rhi0 = pyart.io.read(rhi_0['cf'])
    rhi20 = pyart.io.read(rhi_20['cf'])
    rhi90 = pyart.io.read(rhi_90['cf'])
    rhi120 = pyart.io.read(rhi_120['cf'])
    colorbar_label = '(dBZ)'

    display = pyart.graph.RadarDisplay(rhi0)
#    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    display.plot('filtered_refectivity', 0, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax1.set_ylabel('hgt',fontsize=18)
    az1=rhi_0['az']
    T0 =rhi_0['time']
    delT0 = 0
    ax1.set_title(f'RHI0 T0:{T0:%H%M%S} delT:{delT0} AZ:{az1:.1f}',fontsize=22)

    ax2 = fig.add_subplot(412)
    display = pyart.graph.RadarDisplay(rhi20)
    display.plot('filtered_refectivity', 0, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax2.set_ylabel('hgt',fontsize=18)
    az2=rhi_20['az']
    T1 =rhi_20['time']
    delT01 = (T1-T0).seconds
    ax2.set_title(f'RHI2 T1:{T1:%H%M%S} delT:{delT01} AZ:{az2:.1f}',fontsize=22)



    ax3 = fig.add_subplot(413)
    display = pyart.graph.RadarDisplay(rhi90)
    display.plot('filtered_refectivity', 0, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax3.set_ylabel('hgt',fontsize=18)

    az3=rhi_90['az']
    T2 =rhi_90['time']
    delT02 = (T2-T0).seconds
    ax3.set_title(f'RHI3 T2:{T2:%H%M%S} delT:{delT02} AZ:{az3:.1f}',fontsize=22)


    ax4 = fig.add_subplot(414)
    display = pyart.graph.RadarDisplay(rhi120)
    display.plot('filtered_refectivity', 0, vmin=0, vmax=80, cmap=configdat.temp_cmap,colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax4.set_ylabel('hgt',fontsize=18)
    az4=rhi_120['az']
    T3 =rhi_120['time']
    delT03 = (T3-T0).seconds
    ax4.set_title(f'RHI4 T3:{T3:%H%M%S} delT:{delT03} AZ:{az4:.1f}',fontsize=22)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    plt.suptitle(f'Date {T0:%Y%m%d}',fontsize=22,y=1.01)

    plt.tight_layout()

    plt.savefig(f"{basedir}CSAPR_RAWRHI_4panel_{T0:%Y%m%d_%H%M%S}_{scannum}_{extra}.png",bbox_inches='tight',facecolor='white',dpi=400)
    
    
def plot_raw_ve(rhi_0,rhi_20, rhi_90,rhi_120,scannum,xlim=[0,30],ylim=[0,15],basedir='./',extra=''):
    fig = plt.figure(figsize=(10, 20))

    rhi0 = pyart.io.read(rhi_0['cf'])
    rhi20 = pyart.io.read(rhi_20['cf'])
    rhi90 = pyart.io.read(rhi_90['cf'])
    rhi120 = pyart.io.read(rhi_120['cf'])
    colorbar_label = 'Radial Velocity'

    display = pyart.graph.RadarDisplay(rhi0)
#    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    display.plot('filtered_CV', 0, vmin=-20, vmax=20, cmap='bwr',colorbar_label=colorbar_label)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax1.set_ylabel('Height (km)',fontsize=18)
    az1=rhi_0['az']
    T0 =rhi_0['time']
    delT0 = 0
    ax1.set_title(f'RHI0 T0:{T0:%H%M%S} delT:{delT0} AZ:{az1:.1f}',fontsize=22)

    ax2 = fig.add_subplot(412)
    display = pyart.graph.RadarDisplay(rhi20)
    display.plot('filtered_CV', 0, vmin=-20, vmax=20, cmap='bwr',colorbar_label=colorbar_label)

    plt.ylim(ylim)
    plt.xlim(xlim)
    ax2.set_ylabel('Height (km)',fontsize=18)
    az2=rhi_20['az']
    T1 =rhi_20['time']
    delT01 = (T1-T0).seconds
    ax2.set_title(f'RHI2 T1:{T1:%H%M%S} delT:{delT01} AZ:{az2:.1f}',fontsize=22)



    ax3 = fig.add_subplot(413)
    display = pyart.graph.RadarDisplay(rhi90)
    display.plot('filtered_CV', 0, vmin=-20, vmax=20, cmap='bwr',colorbar_label=colorbar_label)

    plt.ylim(ylim)
    plt.xlim(xlim)
    ax3.set_ylabel('Height (km)',fontsize=18)

    az3=rhi_90['az']
    T2 =rhi_90['time']
    delT02 = (T2-T0).seconds
    ax3.set_title(f'RHI3 T2:{T2:%H%M%S} delT:{delT02} AZ:{az3:.1f}',fontsize=22)


    ax4 = fig.add_subplot(414)
    display = pyart.graph.RadarDisplay(rhi120)
    display.plot('filtered_CV', 0, vmin=-20, vmax=20, cmap='bwr',colorbar_label=colorbar_label)

    plt.ylim(ylim)
    plt.xlim(xlim)
    ax4.set_ylabel('Height (km)',fontsize=18)
    az4=rhi_120['az']
    T3 =rhi_120['time']
    delT03 = (T3-T0).seconds
    ax4.set_title(f'RHI4 T3:{T3:%H%M%S} delT:{delT03} AZ:{az4:.1f}',fontsize=22)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    plt.suptitle(f'Date {T0:%Y%m%d}',fontsize=22,y=1.01)

    plt.tight_layout()

    plt.savefig(f"{basedir}CSAPR_RAWRHI_4panel_{T0:%Y%m%d_%H%M%S}_CV_{scannum}_{extra}.png",bbox_inches='tight',facecolor='white',dpi=400)
    
    
def get_label_region(dz,thresh,hgt,rng):
    label_mask = generate_binary_structure(2,2)


    inds = np.where(dz>=thresh)
    pf_groups, n_groups = label(dz >= thresh, structure = label_mask)

    pf_groups -= 1
    pf_cent_locs   = center_of_mass(dz >=thresh, pf_groups , np.arange(n_groups))

    print(f'found {n_groups}')
    grps = {}
    areap = 0
    lggrp = 0
    if n_groups >0:
        for i in range(n_groups):
            vals = {}
            area1 =len(np.where(pf_groups==i)[1])
            vals['area']=area1
            #print(f'{i} {area1}')
            l = pf_cent_locs[i]
            #print(pf_cent_locs[i])
            #pf_locs       = [(rng[int(l[1])], hgt[int(l[0])]) for l in pf_cent_locs]
            vals['crng']=rng[int(l[1])]
            vals['hgt']=hgt[int(l[0])]
            grps[i]=vals
            if area1 > areap:
                areap = area1
                lggrp = i
        grps['large'] = lggrp
        print('lggroup',lggrp,areap)
        grps['pfs']=pf_groups
    else:
        print("No region found")
        grps = np.nan
    return grps

def max_dbz_height(dz,zht):
    dzmax = np.nanmax(dz,axis=0)
    dzdum = deepcopy(dz)
    whbad = np.where(np.isnan(dzdum))
    dzdum[whbad]=-9999.
    whhgt = np.nanargmax(dzdum,axis=0)
    zht = zht[whhgt]
#    whhgt = np.argmin(dz,axis=0)

    return zht


def mean_change_dbz(dz,dz2,zht,delT):
    check_dz = np.arange(0,70)
    dum = []
    for i in check_dz:
        ck1 = get_height(dz,zht,i)
        ck2 = get_height(dz2,zht,i)
        diff_ck = (ck2-ck1)/delT*1000.
        dum.append(diff_ck)
    dum = np.array(dum)
    ascent_mean=np.nanmean(dum,axis=0)
    return ascent_mean

def get_echo_data(rhidat):
    heights1 = np.zeros_like(rhidat['dz'])
    for i,r in enumerate(rhidat['x']):
        heights1[:,i] = rhidat['z']

    eth_n10 = get_height(rhidat['dz'],rhidat['z'],-10)
    eth_0 = get_height(rhidat['dz'],rhidat['z'],0)
    eth_10 = get_height(rhidat['dz'],rhidat['z'],10)
    eth_20 = get_height(rhidat['dz'],rhidat['z'],20)
    eth_30 = get_height(rhidat['dz'],rhidat['z'],30)
    eth_35 = get_height(rhidat['dz'],rhidat['z'],35)
    eth_40 = get_height(rhidat['dz'],rhidat['z'],40)
    eth_50 = get_height(rhidat['dz'],rhidat['z'],50)
    eth_55 = get_height(rhidat['dz'],rhidat['z'],55)
    eth_60 = get_height(rhidat['dz'],rhidat['z'],60)

    rhidat['hgts']=heights1
    rhidat['ethn10']=eth_n10
    rhidat['eth_0']=eth_0
    rhidat['eth_10']=eth_10
    rhidat['eth_20']=eth_20
    rhidat['eth_30']=eth_30
    rhidat['eth_35']=eth_35
    rhidat['eth_40']=eth_40
    rhidat['eth_50']=eth_50
    rhidat['eth_55']=eth_55
    rhidat['eth_60']=eth_60
    
    
    zht = max_dbz_height(rhidat['dz'],rhidat['z'])
    rhidat['maxzh']=zht


    return rhidat
    
def get_hgt_diffs(rhi1,rhi2):
    delT = (rhi2['time']-rhi1['time']).seconds
    diff_dz = get_diffs(rhi1['dz'],rhi2['dz'])
    delaz = (rhi2['az']-rhi1['az'])
    #diff_dz_lin = get_diffs_lin(dzslantrhi1,dzslantrhi2)
    diff_ve = get_diffs(rhi1['ve'],rhi2['ve'])

    ascent_rate_eth = (rhi2['ethn10']-rhi1['ethn10'])/delT*1000.
    ascent_rate_0 = (rhi2['eth_0']-rhi1['eth_0'])/delT*1000.
    ascent_rate_10 = (rhi2['eth_10']-rhi1['eth_10'])/delT*1000.
    ascent_rate_20 = (rhi2['eth_20']-rhi1['eth_20'])/delT*1000.
    ascent_rate_30 = (rhi2['eth_30']-rhi1['eth_30'])/delT*1000.
    ascent_rate_35 = (rhi2['eth_35']-rhi1['eth_35'])/delT*1000.
    ascent_rate_40 = (rhi2['eth_40']-rhi1['eth_40'])/delT*1000.
    ascent_rate_50 = (rhi2['eth_50']-rhi1['eth_50'])/delT*1000.
    ascent_rate_55 = (rhi2['eth_55']-rhi1['eth_55'])/delT*1000.
    ascent_rate_60 = (rhi2['eth_60']-rhi1['eth_60'])/delT*1000.

    ascent_rate_maxZ = (rhi2['maxzh']-rhi1['maxzh'])/delT*1000.

    columnDZ = np.nanmax(diff_dz,axis=0)
    columnDZmean = np.nanmean(diff_dz,axis=0)
    mean_dbzs = mean_change_dbz(rhi1['dz'],rhi2['dz'],rhi1['z'],delT)

    data = {'dzdiff':diff_dz,
            'vediff':diff_ve,
            'rhi1dz':rhi1['dz'],
            'rhi2dz':rhi2['dz'],
            'rhi1az':rhi1['az'],
            'rhi2az':rhi2['az'],
            'rate_maxZ':ascent_rate_maxZ,
            'rate_eth':ascent_rate_eth,
            'rate_0':ascent_rate_0,
            'rate_10':ascent_rate_10,
            'rate_20':ascent_rate_20,
            'rate_30':ascent_rate_30,
            'rate_35':ascent_rate_35,
            'rate_40':ascent_rate_40,
            'rate_50':ascent_rate_50,
            'rate_55':ascent_rate_55,
            'rate_60':ascent_rate_60,
            'delT':delT,
            'rhi1y':rhi1['y'],
            'rhi2y':rhi2['y'],
            'rhi1hgt':rhi1['z'],
            'rhi2hgt':rhi2['z'],
            'columnDZ':columnDZ,
            'columnDZmean':columnDZmean,
            'mean_dz':mean_dbzs,
            'delaz':delaz,
            'rhit1':rhi1['time'],
            'rhit2':rhi2['time']
          }
    return data


def get_diffs(dz1,dz2):
    #we need special handling for the regions where one is nan and the other is not
    #print('dz min in get_diffs',np.nanmin(dz1),np.nanmin(dz2))
    
    whdz1 = np.where(np.logical_and(np.isfinite(dz1),np.isnan(dz2)))    
    whdz2 = np.where(np.logical_and(np.isfinite(dz2),np.isnan(dz1)))
 
    #and since log reflectivity can be negative, we need to take the absolute value in these areas for the change.

    diff_dz = dz2-dz1
    diff_dz[whdz1]=np.abs(dz1[whdz1])
    diff_dz[whdz2]=np.abs(dz2[whdz2])
    return diff_dz

#THis is to highlight areas where one profile does not have reflectivity
# def get_diffs(dz1,dz2):
#     #we need special handling for the regions where one is nan and the other is not
#     #print('dz min in get_diffs',np.nanmin(dz1),np.nanmin(dz2))
#     dz1dum = deepcopy(dz1)
#     dz2dum = deepcopy(dz2)
#     # dz1dum[np.isnan(dz1dum)]=-99999.0
#     # dz2dum[np.isnan(dz2dum)]=-99999.0
#     print('resetting diffs')

# #and since log reflectivity can be negative, we need to take the absolute value in these areas for the change.

#     diff_dz = dz2dum-dz1dum
#     #diff_dz[whdz1]=np.abs(dz1[whdz1])
#     # diff_dz[whdz2]=np.abs(dz2[whdz2])
#     return diff_dz

def plot_grid_diffs(diffdat,rhi1slant,rhi2slant,res='100m',basedir='./',xlim=[0,30],ylim=[0,15],extra=''):
    fig, ax = plt.subplots(3,3,figsize=(21,19))
    axf = ax.flatten()

    f = 0
    c = axf[f].pcolormesh(diffdat['rhi1y'],diffdat['rhi1hgt'],diffdat['rhi1dz'],cmap=configdat.temp_cmap,vmin=0,vmax=80)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat['rhi1y'],diffdat['rhi1hgt'],diffdat['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    az1 = diffdat['rhi1az']
    rhit1 = diffdat['rhit1']
    axf[f].set_title(f'RHI1 {az1:.2f} {rhit1:%H%M%S}',fontsize=20)
    #axf[f].set_xlim(30,50)

    f = 1
    c = axf[f].pcolormesh(diffdat['rhi2y'],diffdat['rhi2hgt'],diffdat['rhi2dz'],cmap=configdat.temp_cmap,vmin=0,vmax=80)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat['rhi1y'],diffdat['rhi1hgt'],diffdat['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    az2 = diffdat['rhi2az']
    rhit2 = diffdat['rhit2']
    axf[f].set_title(f'RHI2 {az2:.2f} {rhit2:%H%M%S}',fontsize=20)
    #axf[f].set_xlim(30,50)

    f = 2
    c = axf[f].pcolormesh(diffdat['rhi1y'],diffdat['rhi1hgt'],diffdat['dzdiff'],cmap='bwr',vmin=-20,vmax=20)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat['rhi1y'],diffdat['rhi1hgt'],diffdat['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    dela = diffdat['delaz']
    delT = diffdat['delT']
    axf[f].set_title(f'Diff delaz:{dela:.2f} delT: {delT}',fontsize=20)
    
    for a in axf:
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.grid()
        a.set_ylabel('Height (km)')
        a.set_xlabel('Distance (km)')
    plt.tight_layout()
    
    plt.savefig(f"{basedir}CSAPR_GRIDRHI_diffs_rhi1{rhit1:%Y%m%d_%H%M%S}-rhi2{rhit2:%Y%m%d_%H%M%S}_{delT}_res{res}_{extra}.png",bbox_inches='tight',facecolor='white',dpi=400)

def plot_grid_diffs_nine(diffdat12,diffdat13,diffdat14,rhi1slant,rhi2slant,rhi3slant,rhi4slant,scannum,res='100m',basedir='./',xlim=[0,30],ylim=[0,15],extra=''):
    fig, ax = plt.subplots(3,3,figsize=(21,19))
    axf = ax.flatten()

    for f,i in enumerate([0,1,2]):
        c = axf[f].pcolormesh(diffdat12['rhi1y'],diffdat12['rhi1hgt'],diffdat12['rhi1dz'],cmap=configdat.temp_cmap,vmin=0,vmax=80)
        #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

        axf[f].contour(diffdat12['rhi1y'],diffdat12['rhi1hgt'],diffdat12['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
        cb = plt.colorbar(c,ax=axf[f])
        az1 = diffdat12['rhi1az']
        rhit1 = diffdat12['rhit1']
        axf[f].set_title(f'RHI1 {az1:.2f} {rhit1:%H%M%S}',fontsize=20)
        #axf[f].set_xlim(30,50)

    f = 3
    c = axf[f].pcolormesh(diffdat12['rhi2y'],diffdat12['rhi2hgt'],diffdat12['rhi2dz'],cmap=configdat.temp_cmap,vmin=0,vmax=80)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat12['rhi1y'],diffdat12['rhi1hgt'],diffdat12['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    az2 = diffdat12['rhi2az']
    rhit2 = diffdat12['rhit2']
    axf[f].set_title(f'RHI2 {az2:.2f} {rhit2:%H%M%S}',fontsize=20)
    #axf[f].set_xlim(30,50)

    f = 4
    c = axf[f].pcolormesh(diffdat13['rhi2y'],diffdat13['rhi2hgt'],diffdat13['rhi2dz'],cmap=configdat.temp_cmap,vmin=0,vmax=80)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat13['rhi1y'],diffdat13['rhi1hgt'],diffdat13['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    az2 = diffdat13['rhi2az']
    rhit2 = diffdat13['rhit2']
    axf[f].set_title(f'RHI2 {az2:.2f} {rhit2:%H%M%S}',fontsize=20)
    #axf[f].set_xlim(30,50)


    f = 5
    c = axf[f].pcolormesh(diffdat14['rhi2y'],diffdat14['rhi2hgt'],diffdat14['rhi2dz'],cmap=configdat.temp_cmap,vmin=0,vmax=80)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat14['rhi1y'],diffdat14['rhi1hgt'],diffdat14['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    az2 = diffdat14['rhi2az']
    rhit2 = diffdat14['rhit2']
    axf[f].set_title(f'RHI2 {az2:.2f} {rhit2:%H%M%S}',fontsize=20)
    #axf[f].set_xlim(30,50)



    f = 6
    c = axf[f].pcolormesh(diffdat12['rhi1y'],diffdat12['rhi1hgt'],diffdat12['dzdiff'],cmap='bwr',vmin=-20,vmax=20)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat12['rhi1y'],diffdat12['rhi1hgt'],diffdat12['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    dela = diffdat12['delaz']
    delT = diffdat12['delT']
    axf[f].set_title(f'Diff delaz:{dela:.2f} delT: {delT}',fontsize=20)

    f = 7
    c = axf[f].pcolormesh(diffdat13['rhi1y'],diffdat13['rhi1hgt'],diffdat13['dzdiff'],cmap='bwr',vmin=-20,vmax=20)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat13['rhi1y'],diffdat13['rhi1hgt'],diffdat13['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    dela = diffdat13['delaz']
    delT = diffdat13['delT']
    axf[f].set_title(f'Diff delaz:{dela:.2f} delT: {delT}',fontsize=20)
  
    
    
    f = 8
    c = axf[f].pcolormesh(diffdat14['rhi1y'],diffdat14['rhi1hgt'],diffdat14['dzdiff'],cmap='bwr',vmin=-20,vmax=20)
    #axf[f].contour(rng,zrhi1g,diff_dz,levels=[5],colors=['k'],linestyles=['-'])

    axf[f].contour(diffdat14['rhi1y'],diffdat14['rhi1hgt'],diffdat14['rhi1dz'],levels=[-20,-10,0,35,55],colors=['k'],linestyles=['-'])
    cb = plt.colorbar(c,ax=axf[f])
    dela = diffdat14['delaz']
    delT = diffdat14['delT']
    axf[f].set_title(f'Diff delaz:{dela:.2f} delT: {delT}',fontsize=20)
  
    for a in axf:
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.grid()
        a.set_ylabel('Height (km)')
        a.set_xlabel('Distance (km)')
    plt.tight_layout()
    
    plt.savefig(f"{basedir}CSAPR_GRIDRHI_diffs_rhi1{rhit1:%Y%m%d_%H%M%S}_ninepanel_res{res}_{scannum}_{extra}.png",bbox_inches='tight',facecolor='white',dpi=400)

def plot_eth_diffs_nine(diffs12,diffs13,diffs14,rhi1,rhi2,rhi3,rhi4,scannum,res='100m',basedir='./',xlim=[0,30],ylim=[0,15],extra=''):

    fig, ax = plt.subplots(3,3,figsize=(20,12))
    axf = ax.flatten()
    f=0
    axf[f].plot(rhi1['y'],rhi1['ethn10'],color='k',ls='-',label='RHI1 ETH')
    axf[f].plot(rhi2['y'],rhi2['ethn10'],color='k',ls='dotted',label='RHI2 ETH')

    axf[f].plot(rhi1['y'],rhi1['eth_10'],color='blue',ls='-',label='RHI1 10')
    axf[f].plot(rhi2['y'],rhi2['eth_10'],color='blue',ls='dotted',label='RHI2 10')

    
    axf[f].plot(rhi1['y'],rhi1['eth_50'],color='r',ls='-',label='RHI1 50')
    axf[f].plot(rhi2['y'],rhi2['eth_50'],color='r',ls='dotted',label='RHI2 50')

    axf[f].plot(rhi1['y'],rhi1['eth_40'],color='orange',ls='-',label='RHI1 40')
    axf[f].plot(rhi2['y'],rhi2['eth_40'],color='orange',ls='dotted',label='RHI2 40')

    axf[f].plot(rhi1['y'],rhi1['eth_30'],color='goldenrod',ls='-',label='RHI1 30')
    axf[f].plot(rhi2['y'],rhi2['eth_30'],color='goldenrod',ls='dotted',label='RHI2 30')

    axf[f].plot(rhi1['y'],rhi1['eth_20'],color='green',ls='-',label='RHI1 20')
    axf[f].plot(rhi2['y'],rhi2['eth_20'],color='green',ls='dotted',label='RHI2 20')
    delT12 = diffs12['delT']
    delaz12 = diffs12['delaz']

    axf[f].grid()
    axf[f].legend()
    axf[f].set_xlim(xlim)
    axf[f].set_ylim(ylim)
    axf[f].set_title(f'delT:{delT12}',fontsize=20)
    axf[f].set_ylabel('Height (km)',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)


    f=1
    axf[f].plot(rhi1['y'],rhi1['ethn10'],color='k',ls='-',label='RHI1 ETH')
    axf[f].plot(rhi3['y'],rhi3['ethn10'],color='k',ls='dotted',label='RHI2 ETH')

    axf[f].plot(rhi1['y'],rhi1['eth_10'],color='blue',ls='-',label='RHI1 10')
    axf[f].plot(rhi3['y'],rhi3['eth_10'],color='blue',ls='dotted',label='RHI2 10')

    
    axf[f].plot(rhi1['y'],rhi1['eth_50'],color='r',ls='-',label='RHI1 50')
    axf[f].plot(rhi3['y'],rhi3['eth_50'],color='r',ls='dotted',label='RHI2 50')

    axf[f].plot(rhi1['y'],rhi1['eth_40'],color='orange',ls='-',label='RHI1 40')
    axf[f].plot(rhi3['y'],rhi3['eth_40'],color='orange',ls='dotted',label='RHI2 40')

    axf[f].plot(rhi1['y'],rhi1['eth_30'],color='goldenrod',ls='-',label='RHI1 30')
    axf[f].plot(rhi3['y'],rhi3['eth_30'],color='goldenrod',ls='dotted',label='RHI2 30')

    axf[f].plot(rhi1['y'],rhi1['eth_20'],color='green',ls='-',label='RHI1 20')
    axf[f].plot(rhi3['y'],rhi3['eth_20'],color='green',ls='dotted',label='RHI2 20')
    delT13 = diffs13['delT']
    delaz13 = diffs13['delaz']

    axf[f].grid()
    axf[f].legend()
    axf[f].set_xlim(xlim)
    axf[f].set_ylim(ylim)
    axf[f].set_title(f'delT:{delT13}',fontsize=20)
    axf[f].set_ylabel('Height (km)',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)



    f=2
    axf[f].plot(rhi1['y'],rhi1['ethn10'],color='k',ls='-',label='RHI1 ETH')
    axf[f].plot(rhi4['y'],rhi4['ethn10'],color='k',ls='dotted',label='RHI2 ETH')

    axf[f].plot(rhi1['y'],rhi1['eth_10'],color='blue',ls='-',label='RHI1 10')
    axf[f].plot(rhi4['y'],rhi4['eth_10'],color='blue',ls='dotted',label='RHI2 10')

    
    axf[f].plot(rhi1['y'],rhi1['eth_50'],color='r',ls='-',label='RHI1 50')
    axf[f].plot(rhi4['y'],rhi4['eth_50'],color='r',ls='dotted',label='RHI2 50')

    axf[f].plot(rhi1['y'],rhi1['eth_40'],color='orange',ls='-',label='RHI1 40')
    axf[f].plot(rhi4['y'],rhi4['eth_40'],color='orange',ls='dotted',label='RHI2 40')

    axf[f].plot(rhi1['y'],rhi1['eth_30'],color='goldenrod',ls='-',label='RHI1 30')
    axf[f].plot(rhi4['y'],rhi4['eth_30'],color='goldenrod',ls='dotted',label='RHI2 30')

    axf[f].plot(rhi1['y'],rhi1['eth_20'],color='green',ls='-',label='RHI1 20')
    axf[f].plot(rhi4['y'],rhi4['eth_20'],color='green',ls='dotted',label='RHI2 20')
    delT14 = diffs14['delT']
    delaz14 = diffs14['delaz']

    axf[f].grid()
    axf[f].legend()
    axf[f].set_xlim(xlim)
    axf[f].set_ylim(ylim)
    axf[f].set_title(f'delT:{delT14}',fontsize=20)
    axf[f].set_ylabel('Height (km)',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)


    f=3
    axf[f].plot(diffs12['rhi1y'],diffs12['rate_eth'],color='k',label='ETH')
    axf[f].plot(diffs12['rhi1y'],diffs12['rate_10'],color='blue',label='10 dBZ')
    axf[f].plot(diffs12['rhi1y'],diffs12['rate_20'],color='green',label='20 dBZ')
    axf[f].plot(diffs12['rhi1y'],diffs12['rate_30'],color='goldenrod',label='30 dBZ')
    axf[f].plot(diffs12['rhi1y'],diffs12['rate_40'],color='orange',label='40 dBZ')
    axf[f].plot(diffs12['rhi1y'],diffs12['rate_50'],color='red',label='50 dBZ')


    axf[f].legend()
    axf[f].set_title(f'delT:{delT12}',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)
    axf[f].set_ylabel('Ascent Rate (m/s)',fontsize=20)
    axf[f].grid()
    axf[f].set_xlim(xlim)
       
    f=4
    axf[f].plot(diffs13['rhi1y'],diffs13['rate_eth'],color='k',label='ETH')
    axf[f].plot(diffs13['rhi1y'],diffs13['rate_10'],color='blue',label='10 dBZ')
    axf[f].plot(diffs13['rhi1y'],diffs13['rate_20'],color='green',label='20 dBZ')
    axf[f].plot(diffs13['rhi1y'],diffs13['rate_30'],color='goldenrod',label='30 dBZ')
    axf[f].plot(diffs13['rhi1y'],diffs13['rate_40'],color='orange',label='40 dBZ')
    axf[f].plot(diffs13['rhi1y'],diffs13['rate_50'],color='red',label='50 dBZ')


    axf[f].legend()
    axf[f].set_title(f'delT:{delT13}',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)
    axf[f].set_ylabel('Ascent Rate (m/s)',fontsize=20)
    axf[f].grid()
    axf[f].set_xlim(xlim)
       
    f=5
    axf[f].plot(diffs14['rhi1y'],diffs14['rate_eth'],color='k',label='ETH')
    axf[f].plot(diffs14['rhi1y'],diffs14['rate_10'],color='blue',label='10 dBZ')
    axf[f].plot(diffs14['rhi1y'],diffs14['rate_20'],color='green',label='20 dBZ')
    axf[f].plot(diffs14['rhi1y'],diffs14['rate_30'],color='goldenrod',label='30 dBZ')
    axf[f].plot(diffs14['rhi1y'],diffs14['rate_40'],color='orange',label='40 dBZ')
    axf[f].plot(diffs14['rhi1y'],diffs14['rate_50'],color='red',label='50 dBZ')


    axf[f].legend()
    axf[f].set_title(f'delT:{delT14}',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)
    axf[f].set_ylabel('Ascent Rate (m/s)',fontsize=20)
    axf[f].grid()
    axf[f].set_xlim(xlim)

        
    f=6

    axf[f].scatter(diffs12['rate_50'],diffs12['columnDZ'],color='red',label='columnMax')

    axf[f].scatter(diffs12['rate_40'],diffs12['columnDZ'],color='orange')
    axf[f].scatter(diffs12['rate_30'],diffs12['columnDZ'],color='goldenrod')
    axf[f].scatter(diffs12['rate_20'],diffs12['columnDZ'],color='green')
    axf[f].scatter(diffs12['rate_eth'],diffs12['columnDZ'],color='black')

    axf[f].scatter(diffs12['rate_50'],diffs12['columnDZmean'],color='red',marker='D',edgecolor='k',label='columnMean')

    axf[f].scatter(diffs12['rate_40'],diffs12['columnDZmean'],color='orange',marker='D',edgecolor='k')
    axf[f].scatter(diffs12['rate_30'],diffs12['columnDZmean'],color='goldenrod',marker='D',edgecolor='k')
    axf[f].scatter(diffs12['rate_20'],diffs12['columnDZmean'],color='green',marker='D',edgecolor='k')
    axf[f].scatter(diffs12['rate_eth'],diffs12['columnDZmean'],color='black',marker='D',edgecolor='fuchsia')
    #axf[f].grid()
    axf[f].set_ylabel('Max Column delDBZ ',fontsize=20)
    axf[f].set_xlabel('Column Ascent Rate (m/s)',fontsize=20)
    axf[f].legend()
    rtime1 = rhi1['time']
    rtime2 = rhi2['time']
    
    axf[f].set_title(f"delT:{delT12}",fontsize=20)
    
    
    f=7

    axf[f].scatter(diffs13['rate_50'],diffs13['columnDZ'],color='red',label='columnMax')

    axf[f].scatter(diffs13['rate_40'],diffs13['columnDZ'],color='orange')
    axf[f].scatter(diffs13['rate_30'],diffs13['columnDZ'],color='goldenrod')
    axf[f].scatter(diffs13['rate_20'],diffs13['columnDZ'],color='green')
    axf[f].scatter(diffs13['rate_eth'],diffs13['columnDZ'],color='black')

    axf[f].scatter(diffs13['rate_50'],diffs13['columnDZmean'],color='red',marker='D',edgecolor='k',label='columnMean')

    axf[f].scatter(diffs13['rate_40'],diffs13['columnDZmean'],color='orange',marker='D',edgecolor='k')
    axf[f].scatter(diffs13['rate_30'],diffs13['columnDZmean'],color='goldenrod',marker='D',edgecolor='k')
    axf[f].scatter(diffs13['rate_20'],diffs13['columnDZmean'],color='green',marker='D',edgecolor='k')
    axf[f].scatter(diffs13['rate_eth'],diffs13['columnDZmean'],color='black',marker='D',edgecolor='fuchsia')
    #axf[f].grid()
    axf[f].set_ylabel('Max Column delDBZ ',fontsize=20)
    axf[f].set_xlabel('Column Ascent Rate (m/s)',fontsize=20)
    axf[f].legend()
    rtime1 = rhi1['time']
    rtime2 = rhi2['time']
    
    axf[f].set_title(f"delT:{delT13}",fontsize=20)
    
    
    f=8

    axf[f].scatter(diffs14['rate_50'],diffs14['columnDZ'],color='red',label='columnMax')

    axf[f].scatter(diffs14['rate_40'],diffs14['columnDZ'],color='orange')
    axf[f].scatter(diffs14['rate_30'],diffs14['columnDZ'],color='goldenrod')
    axf[f].scatter(diffs14['rate_20'],diffs14['columnDZ'],color='green')
    axf[f].scatter(diffs14['rate_eth'],diffs14['columnDZ'],color='black')

    axf[f].scatter(diffs14['rate_50'],diffs14['columnDZmean'],color='red',marker='D',edgecolor='k',label='columnMean')

    axf[f].scatter(diffs14['rate_40'],diffs14['columnDZmean'],color='orange',marker='D',edgecolor='k')
    axf[f].scatter(diffs14['rate_30'],diffs14['columnDZmean'],color='goldenrod',marker='D',edgecolor='k')
    axf[f].scatter(diffs14['rate_20'],diffs14['columnDZmean'],color='green',marker='D',edgecolor='k')
    axf[f].scatter(diffs14['rate_eth'],diffs14['columnDZmean'],color='black',marker='D',edgecolor='fuchsia')
    #axf[f].grid()
    axf[f].set_ylabel('Max Column delDBZ ',fontsize=20)
    axf[f].set_xlabel('Column Ascent Rate (m/s)',fontsize=20)
    axf[f].legend()
    rtime1 = rhi1['time']
    rtime2 = rhi2['time']
    
    axf[f].set_title(f"delT:{delT14}",fontsize=20)
    
           
    
    
    plt.suptitle(f"rhi1time:{rtime1:%Y%m%d %H%M%S}",fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{basedir}CSAPR_{rtime1:%Y%m%d%H%M%S}_ninepanel_ETHdiffs_ascentrate_columns_res{res}_{scannum}_{extra}.png',dpi=500,bbox_inches='tight',facecolor='white')


def plot_eth_diffs(diffs,rhi1,rhi2,res='100m',basedir='./',xlim=[0,30],ylim=[0,15],extra=''):

    fig, ax = plt.subplots(3,1,figsize=(10,12))
    axf = ax.flatten()
    f=0
    axf[f].plot(rhi1['y'],rhi1['ethn10'],color='k',ls='-',label='RHI1 ETH')
    axf[f].plot(rhi2['y'],rhi2['ethn10'],color='k',ls='dotted',label='RHI2 ETH')

    axf[f].plot(rhi1['y'],rhi1['eth_10'],color='blue',ls='-',label='RHI1 10')
    axf[f].plot(rhi2['y'],rhi2['eth_10'],color='blue',ls='dotted',label='RHI2 10')


    
    axf[f].plot(rhi1['y'],rhi1['eth_50'],color='r',ls='-',label='RHI1 50')
    axf[f].plot(rhi2['y'],rhi2['eth_50'],color='r',ls='dotted',label='RHI2 50')

    axf[f].plot(rhi1['y'],rhi1['eth_40'],color='orange',ls='-',label='RHI1 40')
    axf[f].plot(rhi2['y'],rhi2['eth_40'],color='orange',ls='dotted',label='RHI2 40')

    axf[f].plot(rhi1['y'],rhi1['eth_30'],color='goldenrod',ls='-',label='RHI1 30')
    axf[f].plot(rhi2['y'],rhi2['eth_30'],color='goldenrod',ls='dotted',label='RHI2 30')

    axf[f].plot(rhi1['y'],rhi1['eth_20'],color='green',ls='-',label='RHI1 20')
    axf[f].plot(rhi2['y'],rhi2['eth_20'],color='green',ls='dotted',label='RHI2 20')

    axf[f].grid()
    axf[f].legend()
    axf[f].set_xlim(xlim)
    axf[f].set_ylim(ylim)
    axf[f].set_ylabel('Height (km)',fontsize=20)
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)

    f=1
    axf[f].plot(diffs['rhi1y'],diffs['rate_eth'],color='k',label='ETH')
    axf[f].plot(diffs['rhi1y'],diffs['rate_10'],color='blue',label='10 dBZ')
    axf[f].plot(diffs['rhi1y'],diffs['rate_20'],color='green',label='20 dBZ')
    axf[f].plot(diffs['rhi1y'],diffs['rate_30'],color='goldenrod',label='30 dBZ')
    axf[f].plot(diffs['rhi1y'],diffs['rate_40'],color='orange',label='40 dBZ')
    axf[f].plot(diffs['rhi1y'],diffs['rate_50'],color='red',label='50 dBZ')



    axf[f].legend()
    axf[f].set_xlabel('Slant from radar (km)',fontsize=20)
    axf[f].set_ylabel('Ascent Rate (m/s)',fontsize=20)
    axf[f].grid()
    axf[f].set_xlim(xlim[0],xlim[1])
#    axf[f].set_ylim(0,15)

#     for a in axf[0:1]:
#         a.set_xlim(30,60)
#  #       a.set_ylim(0,15)
#         a.grid()
        
    f=2

    axf[f].scatter(diffs['rate_50'],diffs['columnDZ'],color='red',label='columnMax')

    axf[f].scatter(diffs['rate_40'],diffs['columnDZ'],color='orange')
    axf[f].scatter(diffs['rate_30'],diffs['columnDZ'],color='goldenrod')
    axf[f].scatter(diffs['rate_20'],diffs['columnDZ'],color='green')
    axf[f].scatter(diffs['rate_eth'],diffs['columnDZ'],color='black')


    axf[f].scatter(diffs['rate_50'],diffs['columnDZmean'],color='red',marker='D',edgecolor='k',label='columnMean')

    axf[f].scatter(diffs['rate_40'],diffs['columnDZmean'],color='orange',marker='D',edgecolor='k')
    axf[f].scatter(diffs['rate_30'],diffs['columnDZmean'],color='goldenrod',marker='D',edgecolor='k')
    axf[f].scatter(diffs['rate_20'],diffs['columnDZmean'],color='green',marker='D',edgecolor='k')
    axf[f].scatter(diffs['rate_eth'],diffs['columnDZmean'],color='black',marker='D',edgecolor='fuchsia')
    #axf[f].grid()
    axf[f].set_ylabel('Max Column delDBZ ',fontsize=20)
    axf[f].set_xlabel('Column Ascent Rate (m/s)',fontsize=20)
    axf[f].legend()
    delT = diffs['delT']
    delaz = diffs['delaz']
    rtime1 = rhi1['time']
    rtime2 = rhi2['time']
    plt.suptitle(f"rhi1time:{rtime1:%Y%m%d %H%M%S} delT: {delT}sec delaz: {delaz:.2f} res:{res}",fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{basedir}CSAPR_100mgridded_{rtime1:%Y%m%d%H%M%S}-{rtime2:%Y%m%d%H%M%S}_diff{delT}_ETHdiffs_ascentrate_columns_res{res}_{extra}.png',dpi=500,bbox_inches='tight',facecolor='white')


#print(eth_2-eth_1)
