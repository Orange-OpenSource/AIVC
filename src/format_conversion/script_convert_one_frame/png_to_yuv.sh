#!/bin/bash

# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>



# Convert a triplet of PNG <src>_<y,u,v>.png to a .yuv file
# and append it to a existing .yuv file <out_file>.
# If needed, <out_file> is created.
# This script will generate some intermediary files
# under <out_dir>, which will be deleted at the end

src=$1
out_file=$2 # With extension, size and everything
out_dir=$3  # For intermediary file only

# Absolute path of the python script to convert pgm to png
convert_img_python_path=$4

ref0_y=${src}_y.png
ref0_u=${src}_u.png
ref0_v=${src}_v.png

# juste pour connaitre la resolution
res=$(python $convert_img_python_path "$ref0_y" ${out_dir}temp.pgm)
wdt=$(echo $res|cut -dx -f1)
hgt=$(echo $res|cut -dx -f2)
wdt2=$((wdt/2))
hgt2=$((hgt/2))

# Creating header for PGM files
stry="P5 $wdt $hgt 255"
struv="P5 $wdt2 $hgt2 255"
LY=${#stry}
LUV=${#struv}
LY=$((LY+1))
LUV=$((LUV+1))
echo  $stry> ${out_dir}headerY.bin
echo  $struv> ${out_dir}headerUV.bin

# Size for Y plane 
sizeY=$((wdt*hgt))
# Size for U and V plane
sizeUV=$((wdt*hgt/4))

block_size=$sizeUV
blockY=($((sizeY/block_size)))
blockUV=($((sizeUV/block_size)))


# le yuv
yuv_out=${out_file}

# png par png : on convertit en pgm puis en yuv
# iflag=skip bytes because we are expressing skip as byte, 
# other parameters are multiples of the U, V size
python $convert_img_python_path ${ref0_y} ${out_dir}temp.pgm &>/dev/null
dd iflag=skip_bytes skip=$LY count=$blockY if=${out_dir}temp.pgm of=${out_dir}tempy.yuv bs=$block_size  &> /dev/null

python $convert_img_python_path ${ref0_u} ${out_dir}temp.pgm &>/dev/null
dd iflag=skip_bytes skip=$LUV count=$blockUV if=${out_dir}temp.pgm of=${out_dir}tempu.yuv bs=$block_size&> /dev/null

python $convert_img_python_path ${ref0_v} ${out_dir}temp.pgm &>/dev/null
dd iflag=skip_bytes skip=$LUV count=$blockUV if=${out_dir}temp.pgm of=${out_dir}tempv.yuv bs=$block_size &> /dev/null

# Cat yuv out so we concatenate the successive frame
cat ${out_dir}tempy.yuv ${out_dir}tempu.yuv ${out_dir}tempv.yuv >> $yuv_out

# Remove intermediary files
rm ${out_dir}tempy.yuv ${out_dir}tempu.yuv ${out_dir}tempv.yuv ${out_dir}temp.pgm ${out_dir}headerY.bin ${out_dir}headerUV.bin
