# Image Processing and Machine Learning Classification Project

## Project Overview
This project is divided into two main parts:
1. Image processing and feature extraction.
2. Machine learning classification using Weka.

The goal of this project is to process images to extract relevant features, then apply machine learning algorithms to classify these images. The first part involves preparing the images by cleaning, resizing, and extracting features. In the second part, we use Weka to test different classifiers and identify the best performing one.

---

## Part 1: Image Processing

The image processing pipeline starts with an RGB image and outputs a set of normalized features ready for classification. Here is the detailed workflow:

1. **RGB Image**: The input image is in RGB format.
2. **Binary Image**: The image is converted into a binary image (black and white).
3. **Noise Removal**: A filtering process is applied to remove noise from the binary image.
4. **Bounding Box**: A bounding box is drawn around the object of interest in the image.
5. **Cropped Image**: The image is cropped to the size of the bounding box, focusing only on the object.
6. **Zoning (X * Y grid)**: The cropped image is divided into a grid with X rows and Y columns.

### Feature Extraction
From the processed image, three sets of features are extracted:

- **Dimensions**: Height, width, and diagonal of the object in the image are calculated and normalized to produce 3 features.
- **Center of Gravity**: The center of gravity of the object is computed and normalized, adding `(X * Y) + 2` features.
- **Black Pixel Density**: The density of black pixels within each zone of the grid is calculated and normalized, adding another `(X * Y) + 2` features.

These extracted features are then saved into an ARFF file for use in the second part of the project.

---

## Part 2: Machine Learning with Weka

In the second part of the project, the ARFF file generated from the image processing step is used to test different machine learning classifiers in Weka. The following algorithms were evaluated:

### Algorithms Tested
- **k-Nearest Neighbors (kNN)**
  - Distance: Euclidean
  - k = 1
- **Random Forest**
  - Max Depth: 20
- **Multilayer Perceptron (MLP)**
  - Learning Rate: 0.1
  - Layers: 6 * 8
  - Training Time: 250 seconds
- **Support Vector Machine (SVM)**
  - Grid Search for parameter tuning

### Performance Results
| Algorithm        | Precision (%) | 
|------------------|---------------|
| kNN              | 92.5714       |
| Random Forest    | 87.1429       |
| MLP              | 94.2857       |
| SVM              | 82.7751       |

The best classifier was the **Multilayer Perceptron (MLP)**, which achieved the following performance metrics:
- **Precision**: 94.2857%
- **Recall**: 94.3456%
- **F-Measure**: 94.0151%

---

## Conclusion
This project demonstrates the process of image feature extraction and classification using machine learning techniques. The combination of image processing and machine learning algorithms allows for efficient and accurate classification of images. In this project, the MLP classifier was identified as the most effective algorithm for this dataset, providing the highest precision, recall, and F-measure.

The next steps could include experimenting with more advanced feature extraction techniques or testing additional machine learning models to further improve classification performance.



## Getting started


To run with CMake :

In projet-tiv-s8, `./mkdir build` then `cd build` then `cmake ../` then `make` then `./Projet-OpenCV_CMake`

If build directory already exist, in projet-tiv-s8/build delete CMakeFiles and CMakeCache.txt and run `cmake ../` then `make` then `./Projet-OpenCV_CMake`

To run with Clion, just make and run

include/features.hpp contains all the functions we use the file contains all the headers of the functions we used to work on the features
