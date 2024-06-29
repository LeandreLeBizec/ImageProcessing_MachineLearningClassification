#include "features.hpp"

Features::Features() {
    mat = vector<vector<double>>();
    className = vector<string>();
}

vector<vector<double>> Features::getFeaturesMatrice(){
    return mat;
}
vector<string> Features::getClassNameVector(){
    return className;
}

Mat Features::loadImage(string path){
    string imName = path;
    Mat image = imread(imName);
    if(image.data == nullptr){
        cerr << "Image not found: "<< imName << endl;
        waitKey(0);
        //system("pause");
        exit(EXIT_FAILURE);
    }
    return image;
}

Mat Features::image_BGRtoBIN(const Mat& im){
    if(im.channels() != 1){
        Mat imGray;
        Mat imBin;
        cvtColor(im, imGray, COLOR_BGR2GRAY);
        threshold(imGray, imBin, 230, 255, THRESH_BINARY);
        return imBin;
    }
    return im;
}

Rect Features::findBoundingBox(const Mat& im){
    Mat imBin = image_BGRtoBIN(im);
    int min_x = imBin.cols;
    int max_x = 0;
    int min_y = imBin.rows;
    int max_y = 0;
    for (int y = 0; y < imBin.rows; y++) {
        for (int x = 0; x < imBin.cols; x++) {
            int pixel = imBin.ptr<unsigned char>(y)[x];
            if(pixel == 0){
                if(x<min_x) min_x = x;
                if(x>max_x) max_x = x;
                if(y<min_y) min_y = y;
                if(y>max_y) max_y = y;
            }
        }
    }
    int width = max_x - min_x;
    int height = max_y - min_y;
    Rect rect(min_x, min_y, width, height);
    return rect;
}

Mat Features::displayBoundingBox(const Mat& im) {
    Mat imBin = image_BGRtoBIN(im);
    Rect boundingBoxRect = findBoundingBox(imBin);
    Mat result = imBin;
    rectangle(result, boundingBoxRect, cv::Scalar(0, 255, 0), 2); // dessine un rectangle vert de 2px de largeur
    return result;
}

bool Features::isEmpty(const Mat &im) {
    Rect boundingBox = findBoundingBox(im);
    if(boundingBox.size().height <= 0 || boundingBox.size().width <= 0){
        return true;
    }
    return false;
}

Mat Features::croppedImage(const Mat& im) {
    Rect boundingBoxRect = findBoundingBox(im);
    Mat croppedImBin = im(boundingBoxRect);
    return croppedImBin;
}


tuple<double,double> Features::findGravityCenter(const Mat& im){
    double x_center =0;
    double y_center = 0;
    double intensity_sum = 0;

    Mat imBin = image_BGRtoBIN(im);
    Mat invIm;
    bitwise_not(imBin, invIm);

    for(int y=0; y<invIm.rows; y++){
        for(int x=0; x<invIm.cols; x++){
            int intensity = invIm.ptr<unsigned char>(y)[x];
            x_center += x*intensity;
            y_center += y*intensity;
            intensity_sum += intensity;
        }
    }
    if(intensity_sum != 0){
        x_center /= intensity_sum;
        y_center /= intensity_sum;
    }
    return make_tuple(x_center, y_center);
}


Mat Features::displayGravityCenter(const Mat& im) {
    auto center = findGravityCenter(im);
    circle(im, Point(get<0>(center), get<1>(center)), 5, cv::Scalar(0, 0, 255), -1);
    return im;
}


Mat Features::removeNoise(const Mat &im) {
    Mat imBin = image_BGRtoBIN(im);
    Mat result = imBin;
    auto center = findGravityCenter(imBin);
    int distMoy=0;
    int blackPixel=0;
    // Récupérer la distance moyenne des pixels noirs au centre de gravité
    for(int y=0; y<result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.ptr<unsigned char>(y)[x] == 0) {
                blackPixel++;
                distMoy += sqrt(pow(x - get<0>(center), 2) + pow(y - get<1>(center), 2));
            }
        }
    }
    distMoy /= blackPixel;
    // Si la distance d'un point > 165% de la distance moyenne => bruit.
    for(int y=0; y<result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            // Si le pixel actuel est noir
            if (result.ptr<unsigned char>(y)[x] == 0) {
                int distance = sqrt(pow(x - get<0>(center), 2) + pow(y - get<1>(center), 2));
                if (distance > distMoy*1.65) {
                    result.ptr<unsigned char>(y)[x] = 255;
                }
            }
        }
    }
    return result;
}

double Features::findPixelDensity(const Mat& im){
    Mat imBin = image_BGRtoBIN(im);
    int zero = 0;
    for(int i = 0; i < imBin.cols; i++){
        for(int j = 0; j < imBin.rows; j++){
            if(((int) imBin.at<unsigned char>(j,i)) == 0){
                zero++;
            }
        }
    }
    return ((double) zero)/(imBin.cols*imBin.rows);
}


vector<Mat> Features::zoning(const Mat& im, int row, int col){
    Mat croppedImBin = croppedImage(im);
    int rowStep = croppedImBin.rows / row;
    int colStep = croppedImBin.cols / col;
    vector<Mat> subImages;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int rowStart = i * rowStep;
            int rowEnd = rowStart + rowStep;
            int colStart = j * colStep;
            int colEnd = colStart + colStep;
            subImages.push_back(croppedImBin(Range(rowStart, rowEnd), Range(colStart, colEnd)));
        }
    }
    return subImages;
}


vector<KeyPoint> Features::findSift(const Mat& im, int size){
    // à utiliser sur une image en niveau de gris
    vector<KeyPoint> keypoints;
    Ptr<SIFT> sift = SIFT::create();
    sift->detect(im, keypoints);
    std::sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
    });
    //Renvoie la valeur de sift la plus intéressante. Les valeurs de SIFT sont associées à des
    //"réponse de détecteur" (mesure de l'intensité et l'étendu de la réponse de l'opérateur
    // de différence gaussienne" donc plus la valeur est élevée plus le point d'intérêt est intéressant
    keypoints.resize(std::min(static_cast<size_t>(size), keypoints.size()));
    return keypoints;
}

HOGDescriptor Features::findHogVector(const Mat& im){
    HOGDescriptor hog; // Créer un objet HOG

    // Définir les paramètres HOG
    hog.winSize = Size(256, 256); // Taille de l'image
    hog.blockSize = Size(64, 64); // Taille des blocks pour le zoning
    hog.blockStride = Size(64, 64); // Distance entre les blocs (zoning flou)
    hog.cellSize = Size(64, 64); // Taille des cellules pour le calcul de gradient
    hog.nbins = 9; // nombre de directions
    vector<float> descriptors;
    hog.compute(im, descriptors); // Calculer les descripteurs HOG

    // Normalisation L2 des descripteurs
    float norm = 0;
    for (int i = 0; i < descriptors.size(); i++) {
        norm += descriptors[i] * descriptors[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < descriptors.size(); i++) {
        descriptors[i] /= norm;
    }
}

void Features::generateFeaturesMatrice(const char *path, int row, int col){
    DIR *dir;
    struct dirent *ent;
    dir = opendir(path);
    if (dir != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (strstr(ent->d_name, ".png") != NULL){
                string imageName = string(ent->d_name);
                string filename = path + imageName;
                //Traitement image
                Mat im = loadImage(filename);
                //si image non vide
                if(!isEmpty(im)){
                    //enlever le bruit
                    Mat imBin = image_BGRtoBIN(im);
                    Mat imBinWithoutNoise = removeNoise(im);
                    Rect bBox = findBoundingBox(imBinWithoutNoise);
                    //Cropper l'image
                    Mat imBox = croppedImage(imBinWithoutNoise);
                    if(!isEmpty(imBox)){
                        // centre de gravité
                        auto center = findGravityCenter(imBox);
                        //densité
                        auto density = findPixelDensity(imBox);
                        //zoning
                        auto zoningVector = zoning(imBox,row,col);
                        regex pattern("([^0-9]*)_[0-9].*");
                        smatch match;
                        if (regex_search(imageName, match, pattern)) {
                            className.push_back(match[1].str());
                        } else {
                            cout << "No match found" << endl;
                        }
                        vector<double> vec;
                        vec.push_back(bBox.height);
                        vec.push_back(bBox.width);
                        vec.push_back(sqrt(pow(bBox.height,2)+pow(bBox.width,2)));
                        vec.push_back(get<0>(center));
                        vec.push_back(get<1>(center));
                        vec.push_back(density);
                        for(const Mat& m : zoningVector){
                            //features du zoning
                            auto mCenter = findGravityCenter(m);
                            auto mDensity = findPixelDensity(m);
                            vec.push_back(get<0>(mCenter));
                            vec.push_back(get<1>(mCenter));
                            vec.push_back(mDensity);
                        }
                        mat.push_back(vec);
                    }
                }
            }
        }
    }
}


void Features::normalize(){
    vector<double> min;
    min.assign(mat[0].size(),1);
    vector<double> max;
    max.assign(mat[0].size(),0);
    for(int i = 0; i < mat.size(); i++){
        for(int j = 0; j < mat[i].size(); j++){
            if(mat[i][j] < min[j]) min[j] = mat[i][j];
            if(mat[i][j] > max[j]) max[j] = mat[i][j];
        }
    }
    for(int i = 0; i < mat.size(); i++){
        for(int j = 0; j < mat[i].size(); j++){
            mat[i][j] = (mat[i][j]-min[j])/(max[j]-min[j]);
        }
    }
}


void Features::exportArff(string arffName, int row, int col){
    ofstream file("../" + arffName + ".arff");
    file << "@relation icon\n";
    file << "@attribute 'height' numeric\n";
    file << "@attribute 'width' numeric\n";
    file << "@attribute 'diag' numeric\n";
    file << "@attribute 'gx' numeric\n";
    file << "@attribute 'gy' numeric\n";
    file << "@attribute 'd' numeric\n";
    for(int i = 1; i <= row * col; i++){
        file << "@attribute 'gx" << i << "' numeric\n";
        file << "@attribute 'gy" << i << "' numeric\n";
        file << "@attribute 'd" << i << "' numeric\n";
    }
    file << "@attribute 'class' {accident, bomb, car, casualty, electricity, fire, firebrigade, flood, gas, injury, paramedics, person, police, roadblock}\n";
    file << "@data\n";
    int i = 0;
    for(vector<double> vec : mat){
        for(double d : vec){
            file << d << ",";
        }
        file << className[i] << endl;
        i++;
    }
    file.close();
}


