#!/bin/bash
date=20220916

cd /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo

for file in /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/$date/*chivo*rhi*.nc
do
   echo $file
   python -W ignore write_rotation_angle.py $file
   
   Radx2Grid -f $file -params /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/Radx2Grid.chivo.rhi_incus_reo_oversample_1.5km_temp.cart -outdir /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/GRID_1.5KM/
   Radx2Grid -f $file -params /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/Radx2Grid.chivo.rhi_incus_reo_oversample_3.0km_temp.cart -outdir /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/GRID_3KM/
   Radx2Grid -f $file -params /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/Radx2Grid.chivo.rhi_incus_100m_temp.cart -outdir /rasmussen-scratch2/bdolan/INCUS/TRACER_CASES/$date/chivo/GRID_100M/
done

