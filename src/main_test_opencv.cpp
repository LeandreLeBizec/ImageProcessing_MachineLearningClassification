#include <iostream>

using namespace std;
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
#include "../include/features.hpp"

int main() {

    //Dimension du zoning
    int row = 4;
    int col = 4;

    Features features;
    features.generateFeaturesMatrice("../../BDD_complet/", row, col);
    features.normalize();
    features.exportArff("icon_4*4_1", row, col);
    cout << "Export ended" << endl;

    //termine le programme lorsqu'une touche est frappee
	waitKey(0);
	//system("pause");
	return EXIT_SUCCESS;
}


int mainExemple(){

    Features features;
    //Charge l'image
//    Mat im = features.loadImage("../../BDD_complet/flood_029_05_6_5.png");
    Mat im = features.loadImage("../../BDD_test/fire_5_5_5_1.png");

    // Image en niveau de gris
    Mat imBin = features.image_BGRtoBIN(im);
    imshow("GrayScale image", imBin);

    // Image sans bruit
    Mat imBinWithoutNoise = features.removeNoise(im);
    imshow("GrayScale image without noise", imBinWithoutNoise);

    // Display BoundingBox
    Mat imageWithBoundingBox = features.displayBoundingBox(im);
    imshow("BoundingBox image", imageWithBoundingBox);
    Rect bb = features.findBoundingBox(im);
    cout << bb << endl;

    // Cropped image
    Mat croppedImageBin = features.croppedImage(im);
    imshow("cropped image", croppedImageBin);

    // Gravity center
    Mat imageWithGravityCenter = features.displayGravityCenter(im);
    imshow("Gravity center image", imageWithGravityCenter);

    // Zonning 3*3
    auto subImages = features.zoning(im, 3, 3);
    for(int i =0; i<subImages.size(); i++){
        imshow(to_string(i), subImages[i]);
    }

    //termine le programme lorsqu'une touche est frappee
    waitKey(0);
    //system("pause");
    return EXIT_SUCCESS;
}

int mainBruit() {

    Features features;
    //Charge l'image
    Mat im = features.loadImage("../../BDD_complet/accident_004_08_6_2.png");

    // Image en niveau de gris
    Mat imBin = features.image_BGRtoBIN(im);
    imshow("GrayScale image", imBin);

    // Image sans bruit
    Mat imBinWithoutNoise = features.removeNoise(im);
    imshow("GrayScale image without noise", imBinWithoutNoise);

    // Test 2
    Mat im2 = features.loadImage("../../BDD_complet/accident_005_03_6_2.png");
    Mat imBin2 = features.image_BGRtoBIN(im2);
    imshow("GrayScale image 2", imBin2);
    Mat imBinWithoutNoise2 = features.removeNoise(im2);
    imshow("GrayScale image without noise 2", imBinWithoutNoise2);

    // Test 3
    Mat im3 = features.loadImage("../../BDD_complet/accident_004_13_6_2.png");
    Mat imBin3 = features.image_BGRtoBIN(im3);
    imshow("GrayScale image 3", imBin3);
    Mat imBinWithoutNoise3 = features.removeNoise(im3);
    imshow("GrayScale image without noise 3", imBinWithoutNoise3);

    // Test 4
    Mat im4 = features.loadImage("../../BDD_complet/accident_005_18_6_2.png");
    Mat imBin4 = features.image_BGRtoBIN(im4);
    imshow("GrayScale image 4", imBin4);
    Mat imBinWithoutNoise4 = features.removeNoise(im4);
    imshow("GrayScale image without noise 4", imBinWithoutNoise4);

    // Test 5
    Mat im5 = features.loadImage("../../BDD_complet/bomb_003_18_6_2.png");
    Mat imBin5 = features.image_BGRtoBIN(im5);
    imshow("GrayScale image 5", imBin5);
    Mat imBinWithoutNoise5 = features.removeNoise(im5);
    imshow("GrayScale image without noise 5", imBinWithoutNoise5);

    // Test 6
    Mat im6 = features.loadImage("../../BDD_complet/bomb_003_21_6_2.png");
    Mat imBin6 = features.image_BGRtoBIN(im6);
    imshow("GrayScale image 6", imBin6);
    Mat imBinWithoutNoise6 = features.removeNoise(im6);
    imshow("GrayScale image without noise 6", imBinWithoutNoise6);


    //termine le programme lorsqu'une touche est frappee
    waitKey(0);
    //system("pause");
    return EXIT_SUCCESS;
}

int mainSIFT(){
    Features features;
//    Mat im = features.loadImage("../../BDD_complet/accident_000_04_7_2.png");
    Mat im = features.loadImage("../../BDD_complet/accident_000_10_2_3.png");
//    Mat im = features.loadImage("../../BDD_complet/bomb_029_19_3_2.png");
//    Mat im = features.loadImage("../../BDD_complet/flood_029_05_6_4.png");
    Mat imBin = features.image_BGRtoBIN(im);
    Mat imGray;
    cvtColor(im, imGray, COLOR_BGR2GRAY);
    imwrite("../image_output1.png", imGray);
    Mat imBinWithoutNoise = features.removeNoise(imBin);
    Rect boundingBoxRect = features.findBoundingBox(imBinWithoutNoise);
    Mat imGrayCropped = imGray(boundingBoxRect);
    vector<KeyPoint> res = features.findSift(imGrayCropped, 10);
    Mat output;
    drawKeypoints(imGrayCropped, res, output);
    cout << res.size() << endl;
    for(KeyPoint k : res){
        cout << k.pt.x << endl;
        cout << k.pt.y << endl;
        cout << k.size << endl;
        cout << k.angle << endl;
    }

    // Afficher l'image avec les keypoints
    imshow("Image avec keypoints", output);

    //termine le programme lorsqu'une touche est frappee
    waitKey(0);
    //system("pause");
    return EXIT_SUCCESS;
}


