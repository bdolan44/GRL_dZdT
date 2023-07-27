import re
import glob
import sys
import os
import pyart

from pathlib import Path

date = '20220916'
radnam = 'chivo'
filename = f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{radnam}/Radx2Grid.{radnam}.rhi_incus_reo_oversample_1.5km.cart'
ofilename =f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{radnam}/Radx2Grid.{radnam}.rhi_incus_reo_oversample_1.5km_temp.cart'

# filename = '/rasmussen-scratch2/bdolan/INCUS/20220622/Radx2Grid.CSAPR.20220622_rhi_incus_reo_oversample_3km.cart'
# ofilename = '/rasmussen-scratch2/bdolan/INCUS/20220622/Radx2Grid.CSAPR.20220622_rhi_incus_reo_oversample_3km_temp.cart'

file  = sys.argv[1]
print('Got file:',file)


radar = pyart.io.read(file)
az = radar.fixed_angle['data'][0]
print('Az is: ',az)
#ss = Path(file).name[0:10]
rnum = Path(file).name[26:30]
rhinum = Path(file).name.split('_')[4].split('.')[0]
ss = Path(file).name.split('_')[2]
print('ss is:',ss)
print('rhis :',rhinum)
namup = f'{rhinum}_{az:.1f}_'

pattern = re.compile("qqrotqq")
pattern2 =re.compile("qqnamqq")
##az = 333.1

with open(ofilename, 'w') as the_file:

    for i, line in enumerate(open(filename)):

        for match in re.finditer(pattern, line):
            line = f"grid_rotation = {az};"
        for match in re.finditer(pattern2, line):
            line = f"netcdf_file_prefix = \"grid.{ss}_{radnam}_over1.5km_{namup}\";"
       
        the_file.write(line)

filename2 = f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{radnam}/Radx2Grid.{radnam}.rhi_incus_100m.cart'
ofilename2 =f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{radnam}/Radx2Grid.{radnam}.rhi_incus_100m_temp.cart'

pattern = re.compile("qqrotqq")
pattern2 =re.compile("qqnamqq")
##az = 333.1

with open(ofilename2, 'w') as the_file2:

    for i, line in enumerate(open(filename2)):

        for match in re.finditer(pattern, line):
            line = f"grid_rotation = {az:.1f};"
        for match in re.finditer(pattern2, line):
            line = f"netcdf_file_prefix = \"grid.{ss}_{radnam}_0.1km_{namup}\";"
       
        the_file2.write(line)

filename3 =f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{radnam}/Radx2Grid.{radnam}.rhi_incus_reo_oversample_3.0km.cart'
ofilename3 =f'/rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/{date}/{radnam}/Radx2Grid.{radnam}.rhi_incus_reo_oversample_3.0km_temp.cart'

pattern = re.compile("qqrotqq")
pattern2 =re.compile("qqnamqq")
##az = 333.1

with open(ofilename3, 'w') as the_file3:

    for i, line in enumerate(open(filename3)):

        for match in re.finditer(pattern, line):
            line = f"grid_rotation = {az:.1f};"
        for match in re.finditer(pattern2, line):
            line = f"netcdf_file_prefix = \"grid.{ss}_{radnam}_3.0km_{namup}\";"
       
        the_file3.write(line)
