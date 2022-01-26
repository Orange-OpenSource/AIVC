---
title: Rate-distortion results
layout: page
excerpt: "Category index"
aside: false
---

## HEVC test sequences

<div style="text-align: justify">
Alongside the results on the <a href="http://clic.compression.cc/2021/">CLIC 2021</a>
dataset provided in the paper, we also assess the performance of AIVC on the
HEVC test sequences. AIVC outperforms HEVC for All Intra coding. As such, these results focus on Random Access (RA) and Low-delay P (LDP) configurations.
<br/>
<br/>
<b>Unlike</b> the results shown in the paper, the anchors here are x264 and x265. The following command is used to code videos with x264 and x265:
</div>

    ```bash
    ffmpeg  -video_size WxH -i raw_video.yuv
            -c:v lib<codec>                               #  <codec>: either x264 or x265
            -pix_fmt yuv420p
            -<codec>-params "keyint=<IP>:min_keyint=<IP>" # <IP>: intra period, set to 32
                                                          # <codec>: either x264 or x265
            -crf <QP>                                     # <QP>: Quality factor
            -preset medium -f rawvideo bitstream.bin
    ```

### Class B: 1080p sequences

<div>
  <table>
    <tr>
      <td><img height="700" src="/assets/rd_results/BasketballDrive_1920x1080_50_420.png"></td>
      <td><img height="700" src="/assets/rd_results/BQTerrace_1920x1080_60_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="/assets/rd_results/Cactus_1920x1080_50_420.png"></td>
      <td><img height="700" src="/assets/rd_results/Kimono_1920x1080_24_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="/assets/rd_results/ParkScene_1920x1080_24_420.png"></td>
    </tr>
</table>
</div>