#!/bin/bash

# Convert the frame <idx_frame> of a .yuv video <src> to 
# 3 PNGs (Y, U & V) named <out_dir>/<idx_frame>_<y,u,v>.png

src=$1
out_dir=$2
idx_frame=$3

# Absolute path of the python script to convert pgm to png
convert_img_python_path=$4

# Parse the yuv file name to retrieve the resolution
# Set underline dash as delimiter
# File name without absolute path i.e. /a/b/c/filename :> filename
filename=$(echo $src | awk -F/ '{print $NF}')
IFS='_' read -a strarr <<< "$filename"
res=${strarr[1]}
IFS='x' read -a strarr <<< "$res"
wdt=${strarr[0]}
hgt=${strarr[1]}

# YUV 420, half the width and half the height
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
blockY=$(($sizeY/$block_size))
blockUV=$(($sizeUV/$block_size))


# yuv --> raw stream
# frame 0 (nothing is skipped, otherwise blockY+2*blockUV per frame)
skip=$(($idx_frame * ($blockY + 2 * $blockUV)))
dd skip=$skip count=$blockY  if=$src of=${out_dir}tempY.raw bs=$block_size &> /dev/null

skip=$((skip+blockY))
dd skip=$skip count=$blockUV  if=$src of=${out_dir}tempU.raw bs=$block_size &> /dev/null

skip=$((skip+blockUV))
dd skip=$skip count=$blockUV  if=$src of=${out_dir}tempV.raw bs=$block_size &> /dev/null

recy=${out_dir}${idx_frame}_y.png
recu=${out_dir}${idx_frame}_u.png
recv=${out_dir}${idx_frame}_v.png

# raw stream --> pgm --> png
cat ${out_dir}headerY.bin ${out_dir}tempY.raw > ${out_dir}rec_y.pgm
python $convert_img_python_path ${out_dir}rec_y.pgm $recy &>/dev/null

cat ${out_dir}headerUV.bin ${out_dir}tempU.raw > ${out_dir}rec_u.pgm
python $convert_img_python_path ${out_dir}rec_u.pgm $recu &>/dev/null

cat ${out_dir}headerUV.bin ${out_dir}tempV.raw > ${out_dir}rec_v.pgm
python $convert_img_python_path ${out_dir}rec_v.pgm $recv &>/dev/null

# Remove intermediary files
rm ${out_dir}headerY.bin ${out_dir}tempY.raw ${out_dir}rec_y.pgm
rm ${out_dir}headerUV.bin ${out_dir}tempU.raw ${out_dir}rec_u.pgm
rm ${out_dir}tempV.raw ${out_dir}rec_v.pgm
