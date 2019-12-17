#ifndef HEADER_H
#define HEADER_H

#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <iostream>
#include <map>
#include <QFile>
#include <QString>
#include <QColor>
#include <QLabel>
#include <QDir>
#include <QDebug>
#include <QStringList>
#include <opencv2/opencv.hpp>

#define MAXLABNUM 30
#define	PNTS_PER_LINE		40
#define	LINES_PER_BLK		10
#define	PTNUM_PER_BLK		(10*40)
#define	BKNUM_PER_FRM		180

struct point2i{
    int x, y;
};

struct point3d{
    double dat[3];
};

struct point4d{
    double dat[4];
};

struct point3fil{
    float x,y,z,i;
    int lab;
};

struct PntCloud {
    point3fil dat[PNTS_PER_LINE][BKNUM_PER_FRM*LINES_PER_BLK];
};

void loadColorTabel(char *filename);
bool getFileList(std::string path, std::vector<std::string> &fileListVec);
void getData(std::string dir, std::vector<std::string> &fileListVec);
void genOGM();
void dilateInstance(int dilatePixel);
void InitVis();
void updateVis();
cv::Scalar getColorForInstance(int id);
void updateInstanceFromPointCloud();
void myMouseCallBack(int event, int x, int y, int flags, void* param);
// 从instance map更新点云信息
void updatePointCloudFromInstance(cv::Mat &_instanceMap, cv::Mat &polyMask, int instanceLabel);
void delPolyMask(cv::Mat &curMap, cv::Mat &polyMask);
void writePointCloud2File(std::string dir, std::vector<std::string> &fileListVec);

#endif // HEADER_H
