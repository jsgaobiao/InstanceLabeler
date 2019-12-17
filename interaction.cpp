#include "header.h"
#include "c_labelstatus.h"

extern cv::Mat visMap;
extern cv::Mat oneFrmMap;   // 单帧的Instance map
extern cv::Mat allFrmMap;  // 记录每个像素的Instance ID
extern C_LabelStatus labelStatus;
extern std::map<int, cv::Scalar> instanceColorMap;
extern std::vector<cv::Point> polyVec;
extern cv::Mat polyMask;
extern int lastMouseX, lastMouseY;
extern std::set<int> instanceFilterSet;    // 过滤一些instance id，不显示他们

void emphasizeInstance(cv::Mat &img, cv::Mat &iMap, int instanceId, int instanceLabel)
{
    if (instanceColorMap.find(instanceId) == instanceColorMap.end()) return;
    cv::Scalar _scalar = getColorForInstance(instanceId);
    for (int i = 0; i < iMap.rows; i ++) {
        for (int j = 0; j < iMap.cols; j ++) {
            if (iMap.at<int>(i, j) == instanceLabel) {
                cv::circle(img, cv::Point(j, i), 2, _scalar, -1);
            }
        }
    }
}

void implementPolyMask(cv::Mat &curMap, cv::Mat &polyMask, int instanceLabel)
{
//    for (int i = 0; i < polyMask.rows; i ++) {
//        for (int j = 0; j < polyMask.cols; j ++) {
//            // 包围框内部区域
//            if (polyMask.at<uchar>(i, j) == 255) {
//                // 当前没有显示的位置，可能是被过滤掉的像素
//                if (visMap.at<cv::Vec3b>(i, j)[0] == 255 &&
//                    visMap.at<cv::Vec3b>(i, j)[1] == 255 &&
//                    visMap.at<cv::Vec3b>(i, j)[2] == 255) continue;
//                // 所有非空像素都标记为相同的instanceLabel
//                if (curMap.at<int>(i, j) != 0) {
//                    curMap.at<int>(i, j) = instanceLabel;
//                }
//            }
//        }
//    }
    // 从instance map更新点云信息
    updatePointCloudFromInstance(curMap, polyMask, instanceLabel);
}

void delPolyMask(cv::Mat &curMap, cv::Mat &polyMask)
{
//    for (int i = 0; i < polyMask.rows; i ++) {
//        for (int j = 0; j < polyMask.cols; j ++) {
//            // 包围框内部区域
//            // 当前没有显示的位置，可能是被过滤掉的像素
//            if (visMap.at<cv::Vec3b>(i, j)[0] == 255 &&
//                visMap.at<cv::Vec3b>(i, j)[1] == 255 &&
//                visMap.at<cv::Vec3b>(i, j)[2] == 255) continue;
//            if (polyMask.at<uchar>(i, j) == 255) {
//                curMap.at<int>(i, j) = -1;
//            }
//        }
//    }
    // 从instance map更新点云信息
    updatePointCloudFromInstance(curMap, polyMask, -1);
}

void myMouseCallBack(int event, int x, int y, int flags, void* param)
{
    cv::Mat curMap;
    cv::Mat tmpVisMap;  // 临时的可视化图片
    if (!(x == -1 && y == -1)) {
        lastMouseX = x; lastMouseY = y;
    }
    // 当前编辑的是累计激光图还是单帧激光图
    if (labelStatus.isShowSingleFrame == 1)
        curMap = oneFrmMap;
    else
        curMap = allFrmMap;

    // 1. Ctrl+左键 选中某个Instance
    if (event == cv::EVENT_LBUTTONDOWN && (flags & cv::EVENT_FLAG_CTRLKEY)) {
        int pv = curMap.at<int>(y, x);
        int instanceId = (pv >> 16) % ((1<<17)-1);
        int lab = pv & 0xFFFF;
        if (instanceId == 0) return;
        labelStatus.curInstanceLabel = pv;
        printf("Selected Instance ID: %d , Label: %d\n", instanceId, lab);
        // 临时加粗可视化被选中的instance
        tmpVisMap = visMap.clone();
        emphasizeInstance(tmpVisMap, curMap, instanceId, pv);
        cv::imshow("InstanceLabeler", tmpVisMap);
    }
    else
    // 1.1 Ctrl+左键(松开)，恢复可视化
    if (event ==cv::EVENT_LBUTTONUP && (flags & cv::EVENT_FLAG_CTRLKEY)) {
        // 取消加粗被选中的instance，恢复正常
        cv::imshow("InstanceLabeler", visMap);
    }
    else
    // 2. 按下鼠标左键，绘制多边形包围框/画刷
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (labelStatus.mode == 0) {    // 多边形模式
            polyVec.push_back(cv::Point(x, y));
            tmpVisMap = visMap.clone();
            cv::polylines(tmpVisMap, polyVec, 1, cv::Scalar(0, 0, 0));  // 绘制多边形
            cv::imshow("InstanceLabeler", tmpVisMap);
        }
        else
        if (labelStatus.mode == 1) {    // 画刷模式
            if (labelStatus.curInstanceLabel != 0) {
                polyMask.setTo(0);
                cv::circle(polyMask, cv::Point(x, y), labelStatus.brushR, cv::Scalar(255), -1);  // 绘制多边形
                implementPolyMask(curMap, polyMask, labelStatus.curInstanceLabel);
                updateInstanceFromPointCloud();
            }
            updateVis();
            cv::imshow("InstanceLabeler", visMap);
        }
    }
    else
    // 3. 按下鼠标右键，完成包围框绘制
    if (event == cv::EVENT_RBUTTONDOWN) {
        if (labelStatus.curInstanceLabel != 0 && polyVec.size() > 0 && labelStatus.mode == 0) {
            polyMask.setTo(0);
            std::vector<std::vector<cv::Point> > fillContAll;
            fillContAll.push_back(polyVec);
            cv::fillPoly(polyMask, fillContAll, cv::Scalar(255));  // 绘制多边形
            implementPolyMask(curMap, polyMask, labelStatus.curInstanceLabel);
            updateInstanceFromPointCloud();
        }
        updateVis();
        polyVec.clear();
        cv::imshow("InstanceLabeler", visMap);
    }
    else
    // 4. 鼠标移动 （如果是画刷模式，画出画刷的大小）
    if (event == cv::EVENT_MOUSEMOVE && labelStatus.mode == 1) {
        // 左键拖拽绘图(只有单帧开启，多帧太卡了)
        if ((flags & cv::EVENT_FLAG_LBUTTON) && labelStatus.isShowSingleFrame == 1) {
            polyMask.setTo(0);
            cv::circle(polyMask, cv::Point(x, y), labelStatus.brushR, cv::Scalar(255), -1);  // 绘制圆形
            implementPolyMask(curMap, polyMask, labelStatus.curInstanceLabel);
            updateInstanceFromPointCloud();
            updateVis();
        }
        tmpVisMap = visMap.clone();
        cv::circle(tmpVisMap, cv::Point(x, y), labelStatus.brushR, cv::Scalar(0,0,0));
        cv::imshow("InstanceLabeler", tmpVisMap);
    }
    else
    // 5. 鼠标滚轮调整画刷大小(画刷模式)
    if (event == cv::EVENT_MOUSEWHEEL && labelStatus.mode == 1) {
        double _v = cv::getMouseWheelDelta(flags);
        labelStatus.brushR = std::max(5., labelStatus.brushR + _v);
        tmpVisMap = visMap.clone();
        cv::circle(tmpVisMap, cv::Point(lastMouseX, lastMouseY), labelStatus.brushR, cv::Scalar(0,0,0));
        cv::imshow("InstanceLabeler", tmpVisMap);
    }
    // 6. Ctrl+右键 将某个Instance加入/移出instanceFilterSet
    if (event == cv::EVENT_RBUTTONDOWN && (flags & cv::EVENT_FLAG_CTRLKEY)) {
        int pv = curMap.at<int>(y, x);
        int instanceId = (pv >> 16) % ((1<<17)-1);
        if (instanceId == 0) return;
        // 加入
        auto itor = instanceFilterSet.find(instanceId);
        if (itor == instanceFilterSet.end()) {
            instanceFilterSet.insert(instanceId);
            printf("Add instance ID: %d to filter set.\n", instanceId);
        }
        else {
            instanceFilterSet.erase(itor);
            printf("Del instance ID: %d from filter set.\n", instanceId);
        }
        // 临时加粗可视化被选中的instance
        tmpVisMap = visMap.clone();
        emphasizeInstance(tmpVisMap, curMap, instanceId, pv);
        cv::imshow("InstanceLabeler", tmpVisMap);
    }
    else
    // 6.1 Ctrl+右键(松开)，恢复可视化
    if (event ==cv::EVENT_RBUTTONUP && (flags & cv::EVENT_FLAG_CTRLKEY)) {
        // 取消加粗被选中的instance，恢复正常
        updateInstanceFromPointCloud();
        updateVis();
    }
}

