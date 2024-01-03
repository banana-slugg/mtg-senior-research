# Convolutional Neural Networks for Color Classification of Magic: The Gathering Cards

This is a backup of the Python scripts I used to complete my Senior Research project for my Bachelor's Degree in Computer Science.

## The Project

The goal of the project was to test several convolutional neural networks on how well they could classify the art of a Magic card into its Color. Several tests were conducted using card art, then tested again with the card art in grayscale to compare the effectiveness.

## Files

`kernels.py` and `kernels_gray.py` contain the kernels (or "filters") used by the main Tensorflow scripts. A handful of common ones were used, with an emphasis on edge detection.

`color_experiment.py` and `gray_experiment.py` are the main Tensorflow scripts. These build the CNNs, run the tests, and export the results to a csv.

the `scryfall_scripts` folder contains a few scripts that were used to download art from [Scryfall](https://scryfall.com/) using the [Scrython](https://github.com/NandaScott/Scrython) library. `scryfall-cherrypicked.py` downloads a smaller subset of images, purely for the use of testing.

## Why?

The original repository is on my university's GitLab, but my access has since been removed.
