#include <QApplication>
#include <header.h>
#include <c_labelstatus.h>

unsigned char colorTable[MAXLABNUM][4];
std::vector<std::string> fileListVec;
std::vector<PntCloud> veloVec;
std::vector<cv::Matx44d>transMatVec;
std::map<int, cv::Scalar> instanceColorMap;
std::vector<point3d> poseVec;
std::vector<cv::Point> polyVec;
double minX,maxX,minY,maxY;
cv::Mat visMap; // 可视化图
cv::Mat allFrmMap;  // 记录每个像素的Instance ID
//std::vector<cv::Mat> frmMapVec;  // 每个单帧的Instance map
cv::Mat oneFrmMap;   // 单帧的Instance map
C_LabelStatus labelStatus;
cv::Mat polyMask;
int totalFrm;
double pixelSize = 0.1;
int startId = 1;
int lastMouseX, lastMouseY;
std::set<int> instanceFilterSet;    // 过滤一些instance id，不显示他们

// 读入point_labeler标注好的数据（labels），将它恢复成range image的格式(CV_32FC3)
// 每个像素保存instance id 和 label， 高16位是instance id, 低16位是label
int main(int argc, char* argv[])
{
    std::printf("[Usage] ./InstanceLabeler [label_dir] [colortable]\n");
    std::printf("For example:\n");
    std::printf("[label_dir](default): data_for_point_labeler (include [labels] and [tag] directories)\n");
    std::printf("[colortable](default): colortable.txt\n");

    if (argc < 3) {
        printf("Args not enough!\n");
        return 0;
    }
    if (!getFileList(std::string(argv[1])+"/tag/", fileListVec)) {
        return 0;
    }
    // 读入colortable
    loadColorTabel(argv[2]);
    // 读入tag/label/velodyne点云/poses.txt
    getData(std::string(argv[1]), fileListVec);

    // 将点云投影到二维栅格地图
    genOGM();

    // 每帧点云进行膨胀，将漏标instance id的激光点吸收到邻近的物体上
    // 参数表示膨胀的像素宽度
    dilateInstance(10);

    // 绘制俯视地图
    visMap = cv::Mat((maxY - minY) / pixelSize, (maxX - minX) / pixelSize, CV_8UC3);
    polyMask = cv::Mat(visMap.rows, visMap.cols, CV_8UC1);
    InitVis();
    printf("Initialization Done.\n");
    // 鼠标事件回调函数
    cv::setMouseCallback("InstanceLabeler", myMouseCallBack);

    // 标注循环
    int waitT = 0;
    while (true) {
        int keyId = cv::waitKey(waitT);
        switch (keyId) {
        case 27: // Esc  清空当前绘制的多边形
            if (!polyVec.empty()) {
                polyVec.clear();
                updateInstanceFromPointCloud();
                updateVis();
            }
            break;
        case 81: // Left Arrow 上一帧
            if (labelStatus.isShowSingleFrame == 0) break;
            labelStatus.curFrame = std::max(labelStatus.curFrame - 1, 0);
            printf("Frame: %d\n", labelStatus.curFrame);
            updateInstanceFromPointCloud();
            updateVis();
            break;
        case 83: // Right Arrow 下一帧
            if (labelStatus.isShowSingleFrame == 0) break;
            labelStatus.curFrame = std::min(labelStatus.curFrame + 1, totalFrm - 1);
            printf("Frame: %d\n", labelStatus.curFrame);
            updateInstanceFromPointCloud();
            updateVis();
            break;
        case 113:  // Q 切换单帧/多帧可视化
            labelStatus.isShowSingleFrame = 1 - labelStatus.isShowSingleFrame;
            updateInstanceFromPointCloud();
            updateVis();
            break;
        case 97: {   // a 新建一个instance id, 并输入类别
            int _id;
            for (_id = startId; _id < (1 << 16) - 1; _id ++) {
                if (instanceColorMap.find(_id) == instanceColorMap.end()) {
                    labelStatus.curInstanceLabel = _id;
                    break;
                }
            }
            getColorForInstance(labelStatus.curInstanceLabel);
            int _lab;
            printf("Enter new instance's label (pedestrian[4] pedestrian2+[5] rider[6] car[7]) : ");
            scanf("%d", &_lab);
            labelStatus.curInstanceLabel = (labelStatus.curInstanceLabel << 16) + _lab;
            printf("New Instance ID: %d , Label: %d\n", _id, _lab);
            break;
        }
        case 100: { // d 删除选中的点云(噪点)
            if (polyVec.size() == 0) break;
            polyMask.setTo(0);
            std::vector<std::vector<cv::Point> > fillContAll;
            fillContAll.push_back(polyVec);
            cv::fillPoly(polyMask, fillContAll, cv::Scalar(255));  // 绘制多边形
            if (labelStatus.isShowSingleFrame == 1) {
                delPolyMask(oneFrmMap, polyMask);
            }
            else {
                delPolyMask(allFrmMap, polyMask);
            }
            updateVis();
            polyVec.clear();
            cv::imshow("InstanceLabeler", visMap);
            break;
        }
        case 112:   // p 切换画刷模式或多边形模式
            labelStatus.mode = 1 - labelStatus.mode;
            polyVec.clear();
            // 切换到画笔模式
            if (labelStatus.mode == 0) updateVis();
            break;
        case 115:  // s 将结果写入原始点云信息
            writePointCloud2File(std::string(argv[1]), fileListVec);
            break;
        case 102:  // f 开启/关闭过滤instance的可视化
            labelStatus.isFiltered = labelStatus.isFiltered ^ 1;
            updateInstanceFromPointCloud();
            updateVis();
            break;
        case 52:   // 4 增加一个行人的instance id
        case 53:   // 5 增加一个行人2+的instance id
        case 54:   // 6 增加一个骑车人的instance id
        case 55:   // 7 增加一个汽车的instance id
            int _id;
            for (_id = startId; _id < (1 << 16) - 1; _id ++) {
                if (instanceColorMap.find(_id) == instanceColorMap.end()) {
                    labelStatus.curInstanceLabel = _id;
                    break;
                }
            }
            getColorForInstance(labelStatus.curInstanceLabel);
            labelStatus.curInstanceLabel = (labelStatus.curInstanceLabel << 16) + (keyId - 48);
            printf("New Instance ID: %d , Label: %d\n", _id, (keyId - 48));
            break;
        default:
            break;
        }
    }

    return 0;
}
