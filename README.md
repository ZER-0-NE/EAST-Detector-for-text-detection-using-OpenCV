# EAST Detector for Text Detection

OpenCV’s EAST(Efficient and Accurate Scene Text Detection ) text detector is a deep learning model, based on a novel architecture and training pattern. It is capable of 
- running at near real-time at 13 FPS on 720p images and 
- obtains state-of-the-art text detection accuracy.

[Link to paper](https://arxiv.org/pdf/1704.03155.pdf)

OpenCV’s text detector implementation of EAST is quite robust, capable of localizing text even when it’s blurred, reflective, or partially obscured.

There are many natural scene text detection challenges that have been described by Celine Mancas-Thillou and Bernard Gosselin in their excellent 2017 paper, [Natural Scene Text Understanding](https://www.tcts.fpms.ac.be/publications/regpapers/2007/VS_cmtbg2007.pdf) below:

- **Image/sensor noise**: Sensor noise from a handheld camera is typically higher than that of a traditional scanner. Additionally, low-priced cameras will typically interpolate the pixels of raw sensors to produce real colors.

- **Viewing angles**: Natural scene text can naturally have viewing angles that are not parallel to the text, making the text harder to recognize.
Blurring: Uncontrolled environments tend to have blur, especially if the end user is utilizing a smartphone that does not have some form of stabilization.

- **Lighting conditions**: We cannot make any assumptions regarding our lighting conditions in natural scene images. It may be near dark, the flash on the camera may be on, or the sun may be shining brightly, saturating the entire image.

- **Resolution**: Not all cameras are created equal — we may be dealing with cameras with sub-par resolution.

- **Non-paper objects**: Most, but not all, paper is not reflective (at least in context of paper you are trying to scan). Text in natural scenes may be reflective, including logos, signs, etc.

- **Non-planar objects**: Consider what happens when you wrap text around a bottle — the text on the surface becomes distorted and deformed. While humans may still be able to easily “detect” and read the text, our algorithms will struggle. We need to be able to handle such use cases.

- **Unknown layout**: We cannot use any a priori information to give our algorithms “clues” as to where the text resides.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### Thanks to [Adrian's Blog](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) for a comprehensive blog on EAST Detector.

## License
[MIT](https://choosealicense.com/licenses/mit/)