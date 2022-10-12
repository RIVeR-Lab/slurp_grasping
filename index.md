---
layout: default
title: Home
nav_order: 1
description: "SLURP!"
permalink: /
---


<html lang="en-US">
<head>
  <meta charset="UTF-8">
  <meta name="viewpoint" content="width=device-width, initial-scale=1.0">
  <!--<link rel="stylesheet" href="style.css"> -->
  <title>SLURP! Spectroscopy of Liquids Using Robot Pre-Touch Sensing</title>
</head>
<body>
  <div class="header-adder">
    <div class="title_set">
      <h1>SLURP! Spectroscopy of Liquids Using Robot Pre-Touch Sensing</h1>
    </div>
    <div class="names">
      <p><strong><a href="https://nhanson.io/">Nathaniel Hanson<sup>1</sup></a>, Wesley Lewis<sup>2</sup>, <a href="https://kpputhuveetil.github.io/" >Kavya Puthuveetil<sup>2</sup></a>, Donelle Furline<sup>1</sup>, <a href="https://akhilpadmanabha.github.io/">Akhil Padmanabha<sup>2</sup></a>, <a href="https://www.tpadir.info/">Taşkin Padir<sup>1</sup> </a>, <a href="https://zackory.com/">Zackory Erickson<sup>2</sup></a></strong></p>
      <p style="text-align: center;"><strong>Northeastern University<sup>1</sup>, Carnegie Mellon<sup>2</sup></strong></p></div>
  </div>

  <div>
    <div style="text-align: center;">
      <iframe width="560" height="315" src="https://www.youtube.com/embed/EFyeUmdglbE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
  </div>
  <h2>Abstract</h2>
  <p>Liquids and granular media are pervasive
  throughout human environments. Their free-flowing nature
  causes people to constrain them into containers. We do so
  with thousands of different types of containers made out
  of different materials with varying sizes, shapes, and colors.
  In this work, we present a state-of-the-art sensing technique
  for robots to perceive what liquid is inside of an unknown
  container. We do so by integrating Visible to Near Infrared
  (VNIR) reflectance spectroscopy into a robot's end effector.
  We introduce a hierarchical model for inferring the material
  classes of both containers and internal contents given spectral
  measurements from two integrated spectrometers. To train
  these inference models, we capture and open source a dataset
  of spectral measurements from over 180 different combinations
  of containers and liquids. Our technique demonstrates over
  85% accuracy in identifying 13 different liquids and granular
  media contained within 13 different containers. The sensitivity
  of our spectral readings allow our model to also identify the
  material composition of the containers themselves with 96%
  accuracy. Overall, VNIR spectroscopy presents a promising
  method to give household robots a general-purpose ability to
  infer the liquids inside of containers, without needing to open
  or manipulate the containers.
  </p>
<div style="text-align: center;">
  <figure>
  
      <img src="images/slurp_cad_rev2.png" alt="Slurp Gripper">
    <figcaption>
    (Left) Real-world assembly of the SLURP gripper. (Right)
Exploded CAD rendering of SLURP gripper paddle showing inte-
grated visible to near infrared spectrometers and active illumination
associated physical gripper assembly.
    </figcaption>
  </figure>
</div>

  <p>
    <a href="https://github.com/Wesleylewis05/Tests">Link to the Github</a>
  </p>
</body>
</html>