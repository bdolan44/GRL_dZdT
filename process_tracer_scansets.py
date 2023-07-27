import sys
import os
sys.path.insert(0, os.path.abspath('/home/bdolan/python/'))
import glob
import pyart
import numpy as np
import matplotlib.pyplot as plt
from csuram import RadarConfig
configdat = RadarConfig.RadarConfig(dz='DBZ')
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy.ndimage import label, generate_binary_structure, center_of_mass
from copy import deepcopy

import pyart
import gzip
import numpy as np
from matplotlib import pyplot as plt
import shutil, os
from datetime import timedelta, datetime
import netCDF4


def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID',
                              dz_field='CZ'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

def process_radar(radar,dznam,venam,ncpnam,ccnam,nyqst,nthresh,ccthresh):

    radar, gatefilter=get_radar_stats(radar,dznam,venam,ncpnam,ccnam,nthresh,ccthresh)
    nf3 = deepcopy(radar.fields[dznam])
    nf3['data'] = np.ma.masked_where(gatefilter.gate_excluded, nf3['data'])
    radar.add_field('filtered_refectivity', nf3, replace_existing=True)

    corr_vel_region = pyart.correct.dealias_region_based(radar,vel_field=venam,gatefilter=gatefilter,nyquist_vel=nyqst)
    radar = add_field_to_radar_object(corr_vel_region['data'],radar,field_name='CV',long_name='Corrected_velocity',
                                   standard_name='Corrected Velocity',dz_field='filtered_refectivity')
    
    return radar

def get_radar_stats(radar,dznam='reflectivity',venam='velocity',ncpnam='normalized_coherent_power',
                    ccnam='cross_correlation_ratio',nthresh=50,ccthresh=0.6):
    az = radar.azimuth['data'][0]
    #print(az)
  

#     rtimedum = pyart.util.datetime_from_radar(radar)
#     rtime = datetime.strptime(str(rtimedum),'%Y-%m-%d %H:%M:%S')
#     
    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_below(ncpnam, nthresh)
    gatefilter.exclude_below(ccnam,ccthresh)

    dz = radar.fields[dznam]['data'][:]
    ncp = radar.fields[ncpnam]['data'][:]
    cc = radar.fields[ccnam]['data'][:]
    dz = radar.fields[dznam]['data'][:]

    whbadat = np.where(np.logical_or(ncp<nthresh,cc<ccthresh))
    dz[whbadat]=np.nan
    maxdz = np.where(dz == np.nanmax(dz))
    radz = radar.gate_z['data']/1000.
    radx = radar.gate_x['data']/1000.
    rady = radar.gate_y['data']/1000.
    rng = radar.range['data']/1000.
    maxz = radz[maxdz]
    maxx = radx[maxdz]
    maxy = radx[maxdz]
    maxr = rng[maxdz[1]]

    whgd =np.where(dz>-10.0)
    eth= np.max(radz[whgd])
    #print(f'Echo top height {eth:.2f}')
    
    label_mask = generate_binary_structure(2,2)

    #calculates contiguous areas where reflectivty is higher than the given threshold
    #assigns every group a unique number, and returns an array where each index is
    #replaced with the number of the group it belongs to, or a zero if there was no data there
    inds = np.where(dz>=-10)
    pf_groups, n_groups = label(dz >= -10, structure = label_mask)
    pf_groups -= 1


    
    
    pf_cent_locs   = center_of_mass(dz >=-10, pf_groups , np.arange(n_groups))
    pf_locs_t     = center_of_mass(dz >= -10, pf_groups , np.arange(n_groups))
    pf_locs       = [(rng[int(l[1])], radz[int(l[0]),int(l[1])]) for l in pf_locs_t]
    radar.add_field_like(dznam,'OBJ',pf_groups)
    
    maxdzgrp = pf_groups[maxdz][0]
    #print('max:',pf_cent_locs[maxdzgrp],maxz,maxr,maxdzgrp)

     
    return radar, gatefilter
            



date = '20220617'
which_radar = 'csapr'
radar_dir = f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{which_radar}/{date}/'
scanset_dir = '/rasmussen-scratch2/bdolan/INCUS/SCANSETS/'
rfiles = sorted(glob.glob(f'{radar_dir}*.nc'))
print(rfiles)#, radar_dir)

if which_radar == 'csapr':
    # venam = 'Vh'
    # dznam = 'Zh'
    # ncpnam = 'SQIh'
    # ccnam = 'RHOHV'

    venam = 'mean_doppler_velocity'
    dznam = 'reflectivity'
    ncpnam = 'normalized_coherent_power'
    ccnam = 'copol_correlation_coeff'

#    ccthresh = 0.8
#    nthresh = 0.4

    ccthresh = 0.7
    nthresh = 0.2


    nyq = 16
    last_scan = 'ppi'
    scans = []
    fils = []
    azs = []
    scansets = {}
    scanset = {}
    scannum = 0
    for i,f in enumerate(rfiles):
        r= pyart.io.read(f)

        curr_scan = r.scan_type
        az =r.azimuth['data'][0]
        rtime = r.time['units'].split(' ')[2]
        rdtime = datetime.strptime(rtime,'%Y-%m-%dT%H:%M:%SZ')
        r = process_radar(r,dznam,venam,ncpnam,ccnam,nyq,nthresh,ccthresh)
        r.beam_width=0.9

       #print(curr_scan)
        if curr_scan == 'sector':
            if scanset:
                scansets[f'{scannum:0>3}']=scanset
                scannum = scannum+1
            
            scanset = {}
            #ppi_scan = f
            scan_name = 'sector'
            #scan_az = az
            # scanset = {'ppi':ftot,
            #            'ppi_time':rdtime,
            #            'scannum':scannum}
            #scannum = scannum+1
            print('Got PPI',rdtime)
        elif last_scan == 'sector' and curr_scan == 'rhi':
            scan_name = 'rhi1'
#            scanset['rhi1']=ftot
            scanset['rhi1_time']=rdtime
            scanset['rhi1_az']=az
            nrhi = 1
            scanset['nrhi']=nrhi
            print('rhi1',rdtime,az)
        elif last_scan == 'rhi1' and curr_scan == 'rhi':
            scan_name = 'rhi2'
#            scanset['rhi2']=ftot
            scanset['rhi2_time']=rdtime
            scanset['rhi2_az']=az
            nrhi = 2
            scanset['nrhi']=nrhi
            print('rhi2',rdtime,az)

        elif last_scan == 'rhi2' and curr_scan == 'rhi':
            scan_name = 'rhi3'
#            scanset['rhi3']=ftot
            scanset['rhi3_time']=rdtime
            scanset['rhi3_az']=az
            nrhi = 3
            scanset['nrhi']=nrhi
            print('rhi3',rdtime,az)

        elif last_scan == 'rhi3' and curr_scan == 'rhi':
            scan_name = 'rhi4'
#            scanset['rhi4']=ftot
            scanset['rhi4_time']=rdtime
            scanset['rhi4_az']=az
            nrhi = 4
            scanset['nrhi']=nrhi
            print('rhi4',rdtime,az)
        elif last_scan == 'rhi4' and curr_scan == 'rhi':
            scan_name = 'rhi5'
#            scanset['rhi5']=ftot
            scanset['rhi5_time']=rdtime
            scanset['rhi5_az']=az
            nrhi = 5
            scanset['nrhi']=nrhi
            print('rhi5',rdtime,az)
        elif last_scan == 'rhi5' and curr_scan == 'rhi':
            scan_name = 'rhi6'
#            scanset['rhi6']=ftot
            scanset['rhi6_time']=rdtime
            scanset['rhi6_az']=az
            nrhi = 6
            scanset['nrhi']=nrhi
            print('rhi6',rdtime,az)
            #scansets.append(scanset)



        else:
            print('Rando!', curr_scan)
            scan_name = curr_scan
            #scansets.append(scanset)
        fname = f'cfrad.{rdtime:%Y%m%d_%H%M%S}_{scannum:0>3}_csapr_{scan_name}.nc'
        ftot = f'{radar_dir}{fname}'
        scanset[scan_name]=ftot
        pyart.io.write_cfradial(ftot,r,time_reference=True)

             
        last_scan = scan_name
        azs.append(az)
        scansets[f'{scannum:0>3}']=scanset

        #print(i, f, scan_name)


if which_radar == 'chivo':


    venam = 'velocity'
    dznam = 'reflectivity'
    ncpnam = 'radar_echo_classification'
    ccnam = 'cross_correlation_ratio'
    ccthresh = 0.6
    nthresh = 50


    scannum = 7
    scansets = {}
    for i,f in enumerate(rfiles):
        scanset = {}
        radar = pyart.io.read(f)    
        angs1 = radar.fixed_angle['data'][:].data
        print(scannum)
        scanset['scannum']=scannum    
        scanset['nrhi']=len(angs1)
        for j,d in enumerate(angs1):
            rsub = radar.extract_sweeps([j])
            #rtime = rsub.time['units'].split(' ')[2]
            rtime = pyart.util.datetimes_from_radar(rsub)[0]
            print(rtime)
            rdtime = datetime.strptime(str(rtime),'%Y-%m-%d %H:%M:%S')
            nyq =rsub.instrument_parameters['nyquist_velocity']['data'][0]

            rsub = process_radar(rsub,dznam,venam,ncpnam,ccnam,nyq,nthresh,ccthresh)
            rsub.beam_width=0.9

            fname = f'cfrad.{rdtime:%Y%m%d_%H%M%S}_{scannum:0>3}_chivo_rhi{j}.nc'
            ftot = f'{radar_dir}{fname}'
            pyart.io.write_cfradial(ftot,rsub,time_reference=True)
            scanset[f'rhi{j}']=ftot
            scanset[f'rhi{j}_time']=rdtime
            scanset[f'rhi{j}_az']=d
        scansets[f'{scannum:0>3}']=scanset

        scannum = scannum+1
            
        
                        
pickle.dump(scansets,open(f'{scanset_dir}/{which_radar}_{date}_scansets.p','wb'))