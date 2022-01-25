---
title: Visual Examples
feature_text: |
  Visual examples of AIVC behavior
# feature_image: "https://picsum.photos/2560/600?image=873"
# feature_image: assets/diagram/Global_diagram.png
excerpt: "A demo of Markdown and HTML includes"
aside: false  # No about AIVC
---

# For latex

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript"
  src="http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
Use firefox!

Examples are presented on the video sequence *Sports_1080P-6710* from the
[CLIC 2021](http://clic.compression.cc/2021/) dataset.

{% include figure.html image="../assets/diagram/Global_diagram.png" alt="Image with just alt text" %}


## Videos from the paper

The videos presented here are from the Fig. 2 in the paper *AIVC: Artificial
Intelligence for Video Coding*, Ladune *et al.*

<div>
  <table>
    <tr>
      <td>
        <div style="text-align: center">Original video \(\mathbf{x}_t\) </div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/rawframe_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Optical flow \(\mathbf{v}_p\) </div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/vprev_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Optical flow \(\mathbf{v}_f\) </div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/vnext_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Coding mode </div>
        <div style="text-align: center">selection \(\boldsymbol{\alpha}\)</div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/alpha_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Skip mode contribution</div>
        <div style="text-align: center">\((1 - \boldsymbol{\alpha}) \odot \tilde{\mathbf{x}}_t\) </div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/skippart_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Decoded video \(\hat{\mathbf{x}}_t\) </div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/outframe_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>
</div>

We also provide supplementary examples which displays some other quantities at
stake during the coding of a video sequence.

<div>
  <table>
    <tr>
      <td>
        <div style="text-align: center">Bi-directional prediction </div>
        <div style="text-align: center">weighting \(\boldsymbol{\beta}\)</div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/beta_all_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align: center">Temporal prediction</div>
        <div style="text-align: center">\(tilde{x}_t\) </div>
      </td>
      <td>
        <video height="240" autoplay loop>
          <source src="../assets/videos/prediction_even_pad.mp4" type="video/mp4">
        </video>
      </td>
    </tr>
  </table>
</div>

## Conditional coding behavior

Conditional coding plays a key role in AIVC compression performance. In order to
highlight its role, we present some insightful videos based on the separate
synthesis of the analysis and conditioning latent variables.