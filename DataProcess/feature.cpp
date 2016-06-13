#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;
using namespace std;
const double PI = 3.14159;

void filter2d(CImg<unsigned char>& img, CImg<float>& filter) {
    CImg<unsigned char> copyImg(img.width(), img.height(), 1, 1);
    for (int i = 0; i < img.width(); i++)
        for (int j = 0; j < img.height(); j++)
            copyImg(i, j, 0) = img(i, j, 0);
    int woffset = filter.width() / 2;
    int hoffset = filter.height() / 2;
    for (int i = 0; i < img.width(); i++) {
        for (int j = 0; j < img.height(); j++) {
            double sum = 0;
            for (int e = 0; e < filter.width(); e++) {
                for (int f = 0; f < filter.height(); f++) {
                    int x = i + e - woffset;
                    int y = j + f - hoffset;
                    int imgValue = 0;
                    if (x >= 0 && x < img.width() && y >= 0 && y < img.height())
                        imgValue = copyImg(x, y, 0);
                    sum += imgValue * filter(e, f, 0);
                }
            }
            int result = (int)sum;
            img(i, j, 0) = (unsigned char)result;
            img(i, j, 1) = img(i, j, 0);
            img(i, j, 2) = img(i, j, 0);
        }
    }
}

void RGB2GRAY(CImg<unsigned char>& img) {
    for (int i = 0; i < img.width(); i++)
        for (int j = 0; j < img.height(); j++) {
            img(i, j, 0) = (img(i, j, 0) * 38 + img(i, j, 1) * 75 + img(i, j, 2) * 15) >> 7;
            img(i, j, 1) = img(i, j, 0);
            img(i, j, 2) = img(i, j, 0);
        }
}

void getImageNames(std::vector<string>& imgNames, std::vector<string>& points)
{
    freopen("D:\\code\\commicDownload\\top.txt", "r", stdin);
    char temp[30];
    while(gets(temp))
    {
        string img(temp);
        imgNames.push_back(img.substr(img.find_last_of("\t") + 1));
    }
    freopen("D:\\code\\commicDownload\\bottom.txt", "r", stdin);
    while(gets(temp))
    {
        string img(temp);
        imgNames.push_back(img.substr(img.find_last_of("\t") + 1));
    }
}

void generateFeature(std::vector<string>& imgNames, std::vector<string>& points)
{
    freopen("feature.txt", "w", stdout);
    for(int idx = 0; idx < imgNames.size(); idx++)
    {
        CImg<unsigned char> img(imgNames[idx].c_str());
        int height = img.height();
        int width = img.width();
        std::vector<int> columnSums(width - 1, 0);
        std::vector<int> rowSums(height - 1, 0);
        for (int i = 0; i < width - 1; i++)
        {
            for (int j = 0; j < height - 1; j++)
            {
                columnSums[i] += abs(img(i, j, 0) - img(i, j + 1, 0));
                rowSums[j] += abs(img(i, j, 0) - img(i + 1, j, 0));
            }
        }
        cout << imgNames[idx] << "\t";
        for (int i = 0; i < width - 1; i++)
            cout << columnSums[i] << " ";
        for (int i = 0; i < height - 1; i++)
            cout << rowSums[i] << " ";
        cout << points[idx] << endl;
    }
}

int main() {
    CImg<unsigned char> img("D:\\code\\commicDownload\\rankDown\\0_374049_14922_148261_50549677.jpg");
    RGB2GRAY(img);
    img.save("test.jpg");
    /*
    std::vector<string> imgNames;
    std::vector<string> points;
    getImageNames(imgNames, points);
    */
    cout << "over";
    return 0;
}
