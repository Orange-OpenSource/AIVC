---
# title: Visual Examples
# feature_text: |
#   Visual examples of AIVC behavior
# feature_image: "https://picsum.photos/2560/600?image=873"
feature_image: ../assets/feature_images/visualisation.png
excerpt: "Visual examples of AIVC behavior"
aside: false  # No about AIVC
---

<!-- For latex -->
<!-- Load with https! Or it does not work -->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

<div style="text-align: justify">
Some insightful videos about how AIVC compress a video sequence. The examples are presented on the video sequence <i>Sports_1080P-6710</i>, extracted from the <a href="http://clic.compression.cc/2021/">CLIC 2021</a>  dataset.
<br/>
The videos embedded in this page work best in Firefox or Safari. Refresh the page if the videos get out of sync.
</div>
<br/>

{% include figure.html image="../assets/diagram/Global_diagram.png" alt="Image with just alt text" %}
<div style="text-align: center">
<i>Overall diagram of the AIVC codec</i>
</div>
<br/>


## Videos from the paper

<div style="text-align: justify">
The videos presented here are from the Fig. 2 in the paper <i>AIVC: Artificial
Intelligence for Video Coding</i>, Ladune <i>et al.</i>
</div>
<br/>
<div>
  <table>
    <tr>
      <td>
        <div style="text-align: center">Original video $\color{black}{\mathbf{x}_t}$ </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/rawframe_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Optical flow $\color{black}{\mathbf{v}_p}$ </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/vprev_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Optical flow $\color{black}{\mathbf{v}_f}$ </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/vnext_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Coding mode </div>
        <div style="text-align: center">selection $\color{black}{\boldsymbol{\alpha}}$</div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/alpha_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Skip mode contribution</div>
        <div style="text-align: center">$\color{black}{(1 - \boldsymbol{\alpha}) \odot \tilde{\mathbf{x}}_t}$ </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/skippart_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Decoded video $\color{black}{\hat{\mathbf{x}}_t}$ </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/outframe_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>
</div>
<br/>
<div style="text-align: justify">
We also provide supplementary examples which displays some other quantities at
stake during the coding of a video sequence.
</div>
<br/>
<div>
  <table>
    <tr>
      <td>
        <div style="text-align: center">Bi-directional prediction </div>
        <div style="text-align: center">weighting $\color{black}{\boldsymbol{\beta}}$</div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/beta_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Temporal prediction</div>
        <div style="text-align: center">$\color{black}{\tilde{x}_t}$ </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/prediction_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>
</div>

<br/>
## Conditional coding behavior

<div style="text-align: justify">
Conditional coding plays a key role in AIVC compression performance. In order to
better understand its behavior, we present some insightful videos based on the
separate synthesis of the analysis and conditioning MNet latent variables. We'll
have a look at one optical flow $\color{black}{\mathbf{v}_p}$ when it is synthesized from:
</div>

<ul style="padding-left: 50px">
  <li> Analysis latent variable only <i>i.e.</i> no decoder-side info used</li>
  <li> Conditioning latent variable only <i>i.e.</i> not a single bit conveyed</li>
  <li> Both latent variables</li>
</ul>

<div>
  <table>
    <tr>
      <td>
        <div style="text-align: center">Optical flow $\color{black}{\mathbf{v}_p}$</div>
        <div style="text-align: center">Only from conditioning </div>
        <div style="text-align: center">latent variable</div>
        <div style="text-align: center"><b>Decoder-side only!</b></div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/vprev_shortcut_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Optical flow $\color{black}{\mathbf{v}_p}$ </div>
        <div style="text-align: center">Only from analysis</div>
        <div style="text-align: center">latent variable </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/vprev_sent_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Optical flow $\color{black}{\mathbf{v}_p}$ </div>
        <div style="text-align: center">From all latent variables </div>
      </td>
      <td>
        <video height="240" preload="auto" autoplay loop>
          <source src="../assets/videos/vprev_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>
</div>

<br/>
<div style="text-align: justify">
Recall that the conditioning is a decoder-side only transform, so the first
video represents the motion information <b>inferred</b> at the decoder without a
single bit received. Most of the small motions in the background are inferred at
the decoder thanks to the conditioning transform. Yet, the motion of the girl in
the foreground is too complex to be anticipated at the decoder. Thus, the
analysis transform transmits motion information solely for the girl.
</div>
