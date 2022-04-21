# SpectrumColor
A class for conversion of spectral data into CIE XYZ, xyY, and Lab color space values.

## Details
SpectrumColor loads standard observer color matching functions and standard illuminant (if necessary) spectral power distribution in order to convert experimental spectrum data into XYZ tristimulus values. Can further convert to xyY or Lab and calculate a distance metric from the spectrum's Lab to a target Lab point for color matching.

## Usage
Instantiate using constructor by passing spectrum wavelength and intensity data. Alternatively use classmethod .from_file(...) to instantiate using a data file.

## Examples
Can be used with automated formulation and spectroscopy and Bayesian optimization to optimize a dye mixture to color match a target color by minimizing distance in Lab space. An example of an optimization campaign to match Pantone 17-3938 using red, green, and blue aqueous dyes is shown below. Automation was carried out using my [Lab Automation framework](https://github.com/JustinJKwok/Lab-Automation) in conjuction with scikit-optimize Bayesian optimizer.

![dye optimization](docs/dye_opt.png)
