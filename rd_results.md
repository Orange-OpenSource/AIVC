---
# title: Rate-distortion results
layout: page
excerpt: "Category index"
aside: false
feature_image: ../assets/feature_images/visualisation.png
---

<!-- # CLIC 2021 dataset -->

<div style="text-align: justify">
In our paper, AIVC is evaluated against the HEVC Test Model (HM) 16.22 on the <a href="http://clic.compression.cc/2021/">CLIC 2021</a> validation dataset. In order to get comprehensive results, 3 coding configurations are evaluated:
</div>
<ul>
  <li><b>Random Access (RA)</b>: One I frame and One P frame each 32 images, other frames are B frames</li>
  <li><b>Low-delay P (LDP)</b>: One I frame each 32 images, other frames are P frames</li>
  <li><b>All Intra (AI)</b>: All frames are I frames</li>
</ul>

<img height="700" src="../assets/rd_results/rd_clic.png">
<div style="text-align: center">
<i>Results of AIVC vs HM16.22 on the CLIC21 validation set.</i>
</div>
<br/>


# HEVC test sequences

<div style="text-align: justify">
Alongside the results on the <a href="http://clic.compression.cc/2021/">CLIC 2021</a>
dataset provided in the paper, we also assess the performance of AIVC on the
HEVC test sequences. AIVC outperforms HEVC for All Intra coding. As such, these results focus on Random Access (RA) and Low-delay P (LDP) configurations.
<br/>
<br/>
<b>Unlike</b> the results shown in the paper, the anchors here are x264 and x265. The following command is used to code videos with x264 and x265 in the random access configuration:
</div>

```bash
ffmpeg  -video_size WxH -i raw_video.yuv
        -c:v lib<codec>                               #  <codec>: either x264 or x265
        -pix_fmt yuv420p
        -<codec>-params "keyint=<IP>:min_keyint=<IP>" # <IP>: intra period, set to 32
                                                      # <codec>: either x264 or x265
        -crf <QP>                                     # <QP>: Quality factor (22, 27, 32, 37, 42)
        -preset medium -f rawvideo bitstream.bin
```
The low-delay P configuration is obtained by adding the ```-tune zerolatency``` option.

<div style="text-align: justify">
The raw result files are available <a href="https://github.com/Orange-OpenSource/AIVC/tree/gh-page/assets/rd_results">here</a>.
</div>

##### Class B: 1080p sequences

<i>Open images in a new tab to zoom in</i>

<div>
  <table>
    <tr>
      <td><img height="700" src="../assets/rd_results/BasketballDrive_1920x1080_50_420.png"></td>
      <td><img height="700" src="../assets/rd_results/BQTerrace_1920x1080_60_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="../assets/rd_results/Cactus_1920x1080_50_420.png"></td>
      <td><img height="700" src="../assets/rd_results/Kimono_1920x1080_24_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="../assets/rd_results/ParkScene_1920x1080_24_420.png"></td>
    </tr>
</table>
</div>

##### Class C: 480p sequences

<div>
  <table>
    <tr>
      <td><img height="700" src="../assets/rd_results/BasketballDrill_832x480_50_420.png"></td>
      <td><img height="700" src="../assets/rd_results/BQMall_832x480_60_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="../assets/rd_results/PartyScene_832x480_50_420.png"></td>
      <td><img height="700" src="../assets/rd_results/RaceHorses_832x480_30_420.png"></td>
    </tr>
</table>
</div>

##### Class D: 240p sequences

<div>
  <table>
    <tr>
      <td><img height="700" src="../assets/rd_results/BasketballPass_416x240_50_420.png"></td>
      <td><img height="700" src="../assets/rd_results/BlowingBubbles_416x240_50_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="../assets/rd_results/BQSquare_416x240_60_420.png"></td>
      <td><img height="700" src="../assets/rd_results/RaceHorses_416x240_30_420.png"></td>
    </tr>
</table>
</div>

##### Class E: 720p videoconferencing sequences

<div>
  <table>
    <tr>
      <td><img height="700" src="../assets/rd_results/FourPeople_1280x720_60_420.png"></td>
      <td><img height="700" src="../assets/rd_results/Johnny_1280x720_60_420.png"></td>
    </tr>
    <tr>
      <td><img height="700" src="../assets/rd_results/KristenAndSara_1280x720_60_420.png"></td>
    </tr>
</table>
</div>