#ifndef C_LABELSTATUS_H
#define C_LABELSTATUS_H


class C_LabelStatus
{
public:
    C_LabelStatus();
public:
    int mode;
    // 0 多边形选择功能
    // 1 鼠标画刷功能
    int brushR;

    int isShowSingleFrame;
    // 是否只可视化单帧激光

    int curFrame;
    // 当前可视化的点云是第几帧

    int curInstanceLabel;
    // 当前待操作的instance id+label，高16位instance id, 低16位label，0为空

    int isFiltered;
    // 是否过滤某些instance（避免可视化上的重合），需要过滤的instance集合是instanceFilterSet

};

#endif // C_LABELSTATUS_H
