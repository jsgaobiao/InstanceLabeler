#include <header.h>
#include <c_labelstatus.h>

extern unsigned char colorTable[MAXLABNUM][4];
extern std::vector<PntCloud> veloVec;
extern std::vector<cv::Matx44d> transMatVec;
extern int totalFrm;
extern double minX,maxX,minY,maxY;
extern double pixelSize;
extern std::map<int, cv::Scalar> instanceColorMap;
extern std::vector<point3d> poseVec;
extern cv::Mat visMap;
//extern std::vector<cv::Mat> frmMapVec;  // 每个单帧的Instance map
extern cv::Mat oneFrmMap;   // 单帧的Instance map
extern cv::Mat allFrmMap;  // 记录每个像素的Instance ID
extern C_LabelStatus labelStatus;
extern std::set<int> instanceFilterSet;

point3d transMultiply(cv::Matx44d &mat, point4d &p)
{
    point3d ret;
    ret.dat[0] = ret.dat[1] = ret.dat[2] = 0;
    for (int i = 0; i < 3; i ++)
        for (int j = 0; j < 4; j ++) {
            ret.dat[i] += mat(i, j) * p.dat[j];
        }
    return ret;
}
// 根据instance id获取对应的颜色
cv::Scalar getColorForInstance(int id)
{
    auto iter = instanceColorMap.find(id);
    if (iter != instanceColorMap.end()) {
        return iter->second;
    }
    QColor qc = QColor::fromHsl(rand()%360,rand()%256,std::max(rand()%200, 90));
    cv::Scalar ret = cv::Scalar(qc.blue(), qc.green(), qc.red());
    instanceColorMap.insert(std::pair<int, cv::Scalar>(id, ret));
    return ret;
}
// 从instance map更新vismap
void updateVis()
{
    // 判断显示单帧还是多帧图像
    if (labelStatus.isShowSingleFrame == 1) {   // 单帧
        visMap.setTo(255);
        for (int i = 0; i < visMap.rows; i ++) {
            for (int j = 0; j < visMap.cols; j ++) {
                int pv = oneFrmMap.at<int>(i, j);
                if (pv == 0) continue;
                int instanceId = (pv >> 16) % ((1<<17)-1);
                int lab = pv & 0xFFFF;
                // 如果是需要过滤的instance，则不显示
                if (labelStatus.isFiltered) {
                    if (instanceFilterSet.find(instanceId) != instanceFilterSet.end()) {
                        continue;
                    }
                }
                if (lab >= 1 && lab <= 7) {
                    // 没有instance id的点云，按照类别颜色画
                    if (instanceId == 0) {
                            visMap.at<cv::Vec3b>(i, j)[0] = colorTable[lab][2];
                            visMap.at<cv::Vec3b>(i, j)[1] = colorTable[lab][1];
                            visMap.at<cv::Vec3b>(i, j)[2] = colorTable[lab][0];
                    }
                    // 有instance id的点云，每个instance id一个随机颜色
                    else {
                        cv::Scalar _scalar = getColorForInstance(instanceId);
                        visMap.at<cv::Vec3b>(i, j)[0] = _scalar[0];
                        visMap.at<cv::Vec3b>(i, j)[1] = _scalar[1];
                        visMap.at<cv::Vec3b>(i, j)[2] = _scalar[2];
                    }
                }
            }
        }
    }
    else {  // 多帧
        visMap.setTo(255);
        for (int i = 0; i < visMap.rows; i ++) {
            for (int j = 0; j < visMap.cols; j ++) {
                int pv = allFrmMap.at<int>(i, j);
                if (pv == 0) continue;
                int instanceId = (pv >> 16) % ((1<<17)-1);
                int lab = pv & 0xFFFF;
                // 如果是需要过滤的instance，则不显示
                if (labelStatus.isFiltered) {
                    if (instanceFilterSet.find(instanceId) != instanceFilterSet.end()) {
                        continue;
                    }
                }
                if (lab >= 1 && lab <= 7) {
                    // 没有instance id的点云，按照类别颜色画
                    if (instanceId == 0) {
                            visMap.at<cv::Vec3b>(i, j)[0] = colorTable[lab][2];
                            visMap.at<cv::Vec3b>(i, j)[1] = colorTable[lab][1];
                            visMap.at<cv::Vec3b>(i, j)[2] = colorTable[lab][0];
                    }
                    // 有instance id的点云，每个instance id一个随机颜色
                    else {
                        cv::Scalar _scalar = getColorForInstance(instanceId);
                        visMap.at<cv::Vec3b>(i, j)[0] = _scalar[0];
                        visMap.at<cv::Vec3b>(i, j)[1] = _scalar[1];
                        visMap.at<cv::Vec3b>(i, j)[2] = _scalar[2];
                    }
                }
            }
        }
    }
//    cv::Mat tmpVis;
//    cv::resize(visMap, tmpVis, cv::Size(visMap.cols*2, visMap.rows*2));
    cv::imshow("InstanceLabeler", visMap);
}

void updateRangeImage()
{
    int len = PNTS_PER_LINE;
    int wid = BKNUM_PER_FRM*LINES_PER_BLK;
    // 同步更新range image可视化
    cv::Mat labVis(len*10, wid, CV_8UC3);
    cv::Mat labImg(len, wid, CV_8UC3);
    cv::Mat instanceImg(len, wid, CV_8UC3);
    cv::Mat mergeImg(len*2, wid, CV_8UC3);
    labImg.setTo(0);
    instanceImg.setTo(0);
    for (int i = 0; i < len; i ++) {
        for (int j = 0; j < wid; j ++) {
            // 无效激光点
            if (veloVec[labelStatus.curFrame].dat[i][j].i == 0) continue;
            point3fil p = veloVec[labelStatus.curFrame].dat[i][j];
            // instance类别的点云
            int instanceId = int(p.lab >> 16) % ((1<<17)-1);
            int lab = int(p.lab & 0xFFFF);

            if (lab < MAXLABNUM) {
                labImg.at<cv::Vec3b>(i, j)[2] = colorTable[lab][0];
                labImg.at<cv::Vec3b>(i, j)[1] = colorTable[lab][1];
                labImg.at<cv::Vec3b>(i, j)[0] = colorTable[lab][2];
                // 获取instance color
                if (instanceId != 0) {
                    cv::Scalar _s = getColorForInstance(instanceId);
                    instanceImg.at<cv::Vec3b>(i, j)[0] = _s[0];
                    instanceImg.at<cv::Vec3b>(i, j)[1] = _s[1];
                    instanceImg.at<cv::Vec3b>(i, j)[2] = _s[2];
                }
            }
        }
    }
    // --------------------  可视化  -----------------------
    labImg.copyTo(mergeImg(cv::Rect(0, 0, labImg.cols, labImg.rows)));
    instanceImg.copyTo(mergeImg(cv::Rect(0, labImg.rows, instanceImg.cols, instanceImg.rows)));
    cv::resize(mergeImg, labVis, cv::Size(labVis.cols, labVis.rows), 0, 0, cv::INTER_NEAREST);
    cv::putText(labVis, std::to_string(labelStatus.curFrame), cv::Point(30,20), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
    cv::imshow("range image", labVis);
}

// 从instance map更新点云信息
void updatePointCloudFromInstance(cv::Mat &_instanceMap, cv::Mat &polyMask, int instanceLabel)
{
    int len = PNTS_PER_LINE;
    int wid = BKNUM_PER_FRM*LINES_PER_BLK;
    // 判断显示单帧还是多帧图像
    if (labelStatus.isShowSingleFrame == 1) {   // 单帧
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // 无效激光点
                if (veloVec[labelStatus.curFrame].dat[i][j].i == 0) continue;

                point3fil p = veloVec[labelStatus.curFrame].dat[i][j];
                // instance类别的点云
                int instanceId = (p.lab >> 16) % ((1<<17)-1);
                int lab = p.lab & 0xFFFF;

                if (lab >= 1 && lab <= 7) {
                    // 像素坐标
                    cv::Point2d pp((p.x - minX) / pixelSize, (p.y - minY) / pixelSize);
                    if (pp.x < 0 || pp.y < 0 || pp.x >= _instanceMap.cols || pp.y >= _instanceMap.rows) continue;
                    // 只更新包围框内部区域
                    if (polyMask.at<uchar>(pp.y, pp.x) == 255) {
                        // 当前没有显示的位置，可能是被过滤掉的像素
                        if (visMap.at<cv::Vec3b>(pp.y, pp.x)[0] == 255 &&
                            visMap.at<cv::Vec3b>(pp.y, pp.x)[1] == 255 &&
                            visMap.at<cv::Vec3b>(pp.y, pp.x)[2] == 255) continue;
                        // 所有非空像素都标记为相同的instanceLabel
                        if (_instanceMap.at<int>(pp.y, pp.x) != 0) {
                            _instanceMap.at<int>(pp.y, pp.x) = instanceLabel;
                            veloVec[labelStatus.curFrame].dat[i][j].lab = _instanceMap.at<int>(pp.y, pp.x);
                            // 需要删除的点标记为-1
                            if (_instanceMap.at<int>(pp.y, pp.x) == -1)
                                veloVec[labelStatus.curFrame].dat[i][j].lab = 0;
                        }
                    }
                }
            }
        }
    }
    else {  // 多帧
        for (int frm = 0; frm < veloVec.size(); frm ++) {
            for (int i = 0; i < len; i ++) {
                for (int j = 0; j < wid; j ++) {
                    // 无效激光点
                    if (veloVec[frm].dat[i][j].i == 0) continue;
                    point3fil p = veloVec[frm].dat[i][j];
                    // instance类别的点云
                    int instanceId = (p.lab >> 16) % ((1<<17)-1);
                    int lab = p.lab & 0xFFFF;

                    if (lab >= 1 && lab <= 7) {
                        // 像素坐标
                        cv::Point2d pp((p.x - minX) / pixelSize, (p.y - minY) / pixelSize);
                        if (pp.x < 0 || pp.y < 0 || pp.x >= _instanceMap.cols || pp.y >= _instanceMap.rows) continue;
                        // 只更新包围框内部区域
                        if (polyMask.at<uchar>(pp.y, pp.x) == 255) {
                            // 当前没有显示的位置，可能是被过滤掉的像素
                            if (visMap.at<cv::Vec3b>(pp.y, pp.x)[0] == 255 &&
                                visMap.at<cv::Vec3b>(pp.y, pp.x)[1] == 255 &&
                                visMap.at<cv::Vec3b>(pp.y, pp.x)[2] == 255) continue;
                            // 所有非空像素都标记为相同的instanceLabel
                            if (_instanceMap.at<int>(pp.y, pp.x) != 0) {
                                _instanceMap.at<int>(pp.y, pp.x) = instanceLabel;
                                veloVec[frm].dat[i][j].lab = _instanceMap.at<int>(pp.y, pp.x);
                                // 需要删除的点标记为-1
                                if (_instanceMap.at<int>(pp.y, pp.x) == -1)
                                    veloVec[frm].dat[i][j].lab = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    updateRangeImage();
}

// 从点云信息更新instance map
void updateInstanceFromPointCloud()
{
    // 判断显示单帧还是多帧图像
    if (labelStatus.isShowSingleFrame == 1) {   // 单帧
        oneFrmMap.setTo(0);
        int len = PNTS_PER_LINE;
        int wid = BKNUM_PER_FRM*LINES_PER_BLK;
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // 无效激光点
                if (veloVec[labelStatus.curFrame].dat[i][j].i == 0) continue;
                point3fil p = veloVec[labelStatus.curFrame].dat[i][j];
                // 像素坐标
                cv::Point2d pp((p.x - minX) / pixelSize, (p.y - minY) / pixelSize);
                if (pp.x < 0 || pp.y < 0 || pp.x >= oneFrmMap.cols || pp.y >= oneFrmMap.rows) continue;

                // instance类别的点云
                int instanceId = (p.lab >> 16) % ((1<<17)-1);
                int lab = p.lab & 0xFFFF;

                // 如果是需要过滤的instance，则不显示
                if (labelStatus.isFiltered) {
                    if (instanceFilterSet.find(instanceId) != instanceFilterSet.end()) {
                        continue;
                    }
                }
                if (lab >= 1 && lab <= 7) {
                    // 没有instance id的点云，按照类别颜色画
                    if (instanceId == 0) {
                        if (pp.x < 0 || pp.y < 0 || pp.x >= oneFrmMap.cols || pp.y >= oneFrmMap.rows) continue;
                            oneFrmMap.at<int>(pp.y, pp.x) = p.lab;
                    }
                    // 有instance id的点云，每个instance id一个随机颜色
                    else {
                        if (pp.x < 0 || pp.y < 0 || pp.x >= oneFrmMap.cols || pp.y >= oneFrmMap.rows) continue;
                        oneFrmMap.at<int>(pp.y, pp.x) = p.lab;
                    }
                }
            }
        }
    }
    else {  // 多帧
        allFrmMap.setTo(0);
        for (int frm = 0; frm < veloVec.size(); frm ++) {
            int len = PNTS_PER_LINE;
            int wid = BKNUM_PER_FRM*LINES_PER_BLK;
            for (int i = 0; i < len; i ++) {
                for (int j = 0; j < wid; j ++) {
                    // 无效激光点
                    if (veloVec[frm].dat[i][j].i == 0) continue;
                    point3fil p = veloVec[frm].dat[i][j];
                    // 像素坐标
                    cv::Point2d pp((p.x - minX) / pixelSize, (p.y - minY) / pixelSize);
                    if (pp.x < 0 || pp.y < 0 || pp.x >= allFrmMap.cols || pp.y >= allFrmMap.rows) continue;

                    // instance类别的点云
                    int instanceId = (p.lab >> 16) % ((1<<17)-1);
                    int lab = p.lab & 0xFFFF;

                    // 如果是需要过滤的instance，则不显示
                    if (labelStatus.isFiltered) {
                        if (instanceFilterSet.find(instanceId) != instanceFilterSet.end()) {
                            continue;
                        }
                    }
                    if (lab >= 1 && lab <= 7) {
                        // 没有instance id的点云，按照类别颜色画
                        if (instanceId == 0) {
                            if (pp.x < 0 || pp.y < 0 || pp.x >= allFrmMap.cols || pp.y >= allFrmMap.rows) continue;
                                allFrmMap.at<int>(pp.y, pp.x) = p.lab;
                        }
                        // 有instance id的点云，每个instance id一个随机颜色
                        else {
                            if (pp.x < 0 || pp.y < 0 || pp.x >= allFrmMap.cols || pp.y >= allFrmMap.rows) continue;
                            allFrmMap.at<int>(pp.y, pp.x) = p.lab;
                        }
                    }
                }
            }
        }
    }
    updateRangeImage();
}

void InitVis()
{
    visMap.setTo(255);
    oneFrmMap = cv::Mat((maxY - minY) / pixelSize, (maxX - minX) / pixelSize, CV_32SC1);
    allFrmMap = cv::Mat((maxY - minY) / pixelSize, (maxX - minX) / pixelSize, CV_32SC1);
    oneFrmMap.setTo(0);
    allFrmMap.setTo(0);

    // 逐点绘制
    for (int frm = 0; frm < veloVec.size(); frm ++) {
        int len = PNTS_PER_LINE;
        int wid = BKNUM_PER_FRM*LINES_PER_BLK;
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // 无效激光点
                if (veloVec[frm].dat[i][j].i == 0) continue;
                point3fil p = veloVec[frm].dat[i][j];
                // 像素坐标
                cv::Point2d pp((p.x - minX) / pixelSize, (p.y - minY) / pixelSize);
                if (pp.x < 0 || pp.y < 0 || pp.x >= visMap.cols || pp.y >= visMap.rows) continue;

                // instance类别的点云
                int instanceId = (p.lab >> 16) % ((1<<17)-1);
                int lab = p.lab & 0xFFFF;

                if (lab >= 1 && lab <= 7) {
                    // 没有instance id的点云，按照类别颜色画
                    if (instanceId == 0) {
                        if (pp.x < 0 || pp.y < 0 || pp.x >= visMap.cols || pp.y >= visMap.rows) continue;
                        if (visMap.at<cv::Vec3b>(pp.y, pp.x)[0] == 255 && visMap.at<cv::Vec3b>(pp.y, pp.x)[1] == 255 && visMap.at<cv::Vec3b>(pp.y, pp.x)[2] ==255) {
//                            cv::circle(visMap, cv::Point(pp.x, pp.y), 1, cv::Scalar(colorTable[lab][2],colorTable[lab][1],colorTable[lab][0]), -1);
                            visMap.at<cv::Vec3b>(pp.y, pp.x)[0] = colorTable[lab][2];
                            visMap.at<cv::Vec3b>(pp.y, pp.x)[1] = colorTable[lab][1];
                            visMap.at<cv::Vec3b>(pp.y, pp.x)[2] = colorTable[lab][0];
                            allFrmMap.at<int>(pp.y, pp.x) = p.lab;
                        }
                    }
                    // 有instance id的点云，每个instance id一个随机颜色
                    else {
                        if (pp.x < 0 || pp.y < 0 || pp.x >= visMap.cols || pp.y >= visMap.rows) continue;
                        cv::Scalar _scalar = getColorForInstance(instanceId);
//                        cv::circle(visMap, cv::Point(pp.x, pp.y), 1, _scalar, -1);
                        visMap.at<cv::Vec3b>(pp.y, pp.x)[0] = _scalar[0];
                        visMap.at<cv::Vec3b>(pp.y, pp.x)[1] = _scalar[1];
                        visMap.at<cv::Vec3b>(pp.y, pp.x)[2] = _scalar[2];
                        allFrmMap.at<int>(pp.y, pp.x) = p.lab;
                    }
                }
                else { // 非Instance的类别都画成白色
                    // 地面不画
                    if (lab == 22) continue;
                }
            }
        }
    }
    updateRangeImage();
//    cv::Mat tmpVis;
//    cv::resize(visMap, tmpVis, cv::Size(visMap.cols*2, visMap.rows*2));
    cv::imshow("InstanceLabeler", visMap);
//    cv::waitKey(1);
//    cv::imwrite("map.png", visMap);
}

// 将所有点云投影到二维栅格地图
void genOGM()
{
    printf("Converting coordinate...\n");
    minX = minY = 1e20;
    maxX = maxY = -1e20;
    for (int frm = 0; frm < veloVec.size(); frm ++) {
        int wid = BKNUM_PER_FRM*LINES_PER_BLK;
        int len = PNTS_PER_LINE;
        // 计算车体的全局坐标
        point4d curPose;
        curPose.dat[0] = curPose.dat[1] = curPose.dat[2] = 0; curPose.dat[3] = 1;
        point3d newP = transMultiply(transMatVec[frm], curPose);
        poseVec.push_back(newP);
        // 计算每个激光点的全局坐标
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                point3fil tmp = veloVec[frm].dat[i][j];
                if (tmp.i == 0) continue;
                point4d p;
                p.dat[0] = veloVec[frm].dat[i][j].x; p.dat[1] = veloVec[frm].dat[i][j].y;
                p.dat[2] = veloVec[frm].dat[i][j].z; p.dat[3] = 1;
                // 过滤车身上的噪点
//                if (sqrt((p.dat[0]*p.dat[0])*(p.dat[1]*p.dat[1])) < 0.5/* || sqrt((p.dat[0]*p.dat[0])*(p.dat[1]*p.dat[1])) > 60*/) {
//                    veloVec[frm].dat[i][j].i = 0;
//                    continue;
//                }
                // 给点云做坐标变换
                point3d newP = transMultiply(transMatVec[frm], p);
                veloVec[frm].dat[i][j].x = newP.dat[0];
                veloVec[frm].dat[i][j].y = newP.dat[1];
                veloVec[frm].dat[i][j].z = newP.dat[2];

                point3fil ip = veloVec[frm].dat[i][j];
                int instanceId = (ip.lab >> 16) % ((1<<17)-1);
                int lab = ip.lab & 0xFFFF;

                // 人、骑车人、汽车
                if (lab >= 1 && lab <= 7) {
                    // 计算X,Y的最大最小值
                    minX = std::min(minX, newP.dat[0]);
                    minY = std::min(minY, newP.dat[1]);
                    maxX = std::max(maxX, newP.dat[0]);
                    maxY = std::max(maxY, newP.dat[1]);
                }
            }
        }
    }
    // 在图像边界上留白
    minX -= 5; minY -= 5;
    maxX += 5; maxY += 5;
    printf("Convert Over.\n");
}

void dilateInstance(int dilatePixel)
{
    printf("Dilating...\n");
    cv::Mat dilateMap((maxY - minY) / pixelSize, (maxX - minX) / pixelSize, CV_32SC1);
    dilateMap.setTo(0);

    // 逐点膨胀，并将无instance id的点吸收到instance中
    for (int frm = 0; frm < veloVec.size(); frm ++) {
        int len = PNTS_PER_LINE;
        int wid = BKNUM_PER_FRM*LINES_PER_BLK;
        // 将单帧中的instance点进行膨胀
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // 无效激光点
                if (veloVec[frm].dat[i][j].i == 0) continue;
                point3fil p = veloVec[frm].dat[i][j];
                // 像素坐标
                cv::Point2d pp((p.x - minX) / pixelSize,  (p.y - minY) / pixelSize);
                // instance类别的点云
                int instanceId = (p.lab >> 16) % ((1<<17)-1);
                int lab = p.lab & 0xFFFF;

                // 人、骑车人、汽车
                if (lab >= 1 && lab <= 7) {
                    // 有instance id的点,进行膨胀
                    if (pp.x < 0 || dilateMap.cols-pp.y < 0 || pp.x >= dilateMap.cols || dilateMap.cols-pp.y >= dilateMap.rows) continue;
                    int _v = dilateMap.ptr<int>(int(dilateMap.cols-pp.y))[int(pp.x)];
                    if (instanceId != 0 && (_v == 0 || _v == p.lab)) {
                        cv::Scalar _scalar = cv::Scalar(p.lab);
                        cv::circle(dilateMap, cv::Point(pp.x, dilateMap.cols-pp.y), dilatePixel, _scalar, -1);
                    }
                }
            }
        }
    }
//    cv::imshow("debug", dilateMap);
//    cv::waitKey(0);
    for (int frm = 0; frm < veloVec.size(); frm ++) {
        int len = PNTS_PER_LINE;
        int wid = BKNUM_PER_FRM*LINES_PER_BLK;
        // 将无instance id的点归入膨胀后接触到的instance当中
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // 无效激光点
                if (veloVec[frm].dat[i][j].i == 0) continue;
                point3fil p = veloVec[frm].dat[i][j];
                // 像素坐标
                cv::Point2d pp((p.x - minX) / pixelSize,
                               (p.y - minY) / pixelSize);
                if (pp.x < 0 || dilateMap.cols-pp.y < 0 || pp.x >= dilateMap.rows || dilateMap.cols-pp.y >= dilateMap.cols) continue;

                // instance类别的点云
                int instanceId = (p.lab >> 16) % ((1<<17)-1);
                int lab = p.lab & 0xFFFF;

                // 如果Instance ID不为0，跳过
                if (instanceId != 0) continue;

                if (pp.x < 0 || dilateMap.cols-pp.y < 0 || pp.x >= dilateMap.cols || dilateMap.cols-pp.y >= dilateMap.rows) continue;
                int pixelValue = dilateMap.ptr<int>(int(dilateMap.cols-pp.y))[int(pp.x)];
                // 如果该位置没有被膨胀到，不处理    //ptr效率似乎比at高，试一试
                if (pixelValue == 0) continue;
                // 如果该位置点类别和当前点不同，不处理
                if ((pixelValue & 0xFFFF) != lab) continue;

                // 人、骑车人、汽车
                if (lab >= 1 && lab <= 7) {
                    veloVec[frm].dat[i][j].lab = pixelValue;
                }
            }
        }
    }
    printf("Dilate Over.\n");
}
