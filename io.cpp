#include <header.h>

extern unsigned char colorTable[MAXLABNUM][4];
extern std::vector<PntCloud> veloVec;
extern std::vector<cv::Matx44d>transMatVec;
extern int totalFrm;

void loadColorTabel(char *filename)
{
    std::memset (colorTable, 0, MAXLABNUM*4);

    char	i_line[80];
    FILE	*fp;

    fp = std::fopen (filename, "r");
    if (!fp)
        return;

    int n=0;
    while (1) {
        if (fgets (i_line, 80, fp) == NULL)
            break;
        colorTable[n][0] = atoi (strtok (i_line, ",\t\n"));
        colorTable[n][1] = atoi (strtok (NULL, ",\t\n"));
        colorTable[n][2] = atoi (strtok (NULL, ",\t\n"));
        colorTable[n][3] = atoi (strtok (NULL, ",\t\n"));
        n++;
        if (n>=MAXLABNUM)
            break;
    }
    std::fclose (fp);
}

bool getFileList(std::string path, std::vector<std::string> &fileListVec)
{
    QDir *dir = new QDir(QString::fromStdString(path));
    QStringList filter;
    QList<QFileInfo> *fileInfo=new QList<QFileInfo>(dir->entryInfoList(filter));
//    文件数目：fileInfo->count();
//    文件名称：fileInfo->at(i).fileName();
//    文件路径（包含文件名）：fileInfo->at(i).filePath();
    fileListVec.clear();
    if (fileInfo->count() == 0) {
        printf("Empty Input Dir!\n");
        return false;
    }
    totalFrm = fileInfo->count() - 2;
    for (int i = 2; i < fileInfo->count(); i ++) {
        fileListVec.push_back(fileInfo->at(i).fileName().toStdString().substr(0,6));
    }
    return true;
}

void getData(std::string dir, std::vector<std::string> &fileListVec)
{
    int wid = LINES_PER_BLK*BKNUM_PER_FRM;
    int len = PNTS_PER_LINE;
    cv::Mat labVis(len*5, wid, CV_8UC3);
    cv::Mat labImg(len, wid, CV_8UC3);
//    cv::VideoWriter labWriter("labels.avi", CV_FOURCC('M','J','P','G'), 20.0, cv::Size(wid, len));

    FILE* f_pose = fopen(std::string(dir+"/poses.txt").c_str(), "r");

    for (int frm = 0; frm < fileListVec.size(); frm ++) {
        if (frm % 100 == 0) printf("%d\n", frm);
        PntCloud _pc;
        std::ifstream tag_fp(dir+"/tag/"+fileListVec[frm]+".tag", std::ios::binary);
        std::ifstream lab_fp(dir+"/labels/"+fileListVec[frm]+".label", std::ios::binary);
        std::ifstream bin_fp(dir+"/velodyne/"+fileListVec[frm]+".bin", std::ios::binary);

        labImg.setTo(0);
        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // ------------------读入tag---------------------
                char t;
                tag_fp.read(&t, sizeof(char));
                if (t == 1) {
                    // ------------------读入label ----------------
                    int l, instanceId;
                    lab_fp.read((char*)&l, sizeof(int));
                    _pc.dat[i][j].lab = l;
                    // 高16位是instance id
                    instanceId = int(l >> 16) % ((1<<17)-1);
                    // 低16位是类别标签
                    l = int(l & 0xFFFF);
                    // labImg对应位置赋予类别l对应的颜色
                    if (l < MAXLABNUM) {
                        labImg.at<cv::Vec3b>(i, j)[2] = colorTable[l][0];
                        labImg.at<cv::Vec3b>(i, j)[1] = colorTable[l][1];
                        labImg.at<cv::Vec3b>(i, j)[0] = colorTable[l][2];
                    }
                    // -------------------读入点云bin---------------------
                    bin_fp.read((char*)&_pc.dat[i][j].x, sizeof(float));
                    bin_fp.read((char*)&_pc.dat[i][j].y, sizeof(float));
                    bin_fp.read((char*)&_pc.dat[i][j].z, sizeof(float));
                    bin_fp.read((char*)&_pc.dat[i][j].i, sizeof(float));
                }
                else {
                    // 无效激光点位置
                    _pc.dat[i][j].i = 0;
                }
            }
        }
        veloVec.push_back(_pc);
        // ------------------读入poses.txt---------------------
        cv::Matx44d transMat;
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 4; j ++) {
                double v;
                fscanf(f_pose, "%lf", &v);
                transMat(i,j) = v;
            }
        }
        transMat(3,0) = 0; transMat(3,1) = 0;
        transMat(3,2) = 0; transMat(3,3) = 1;
        transMatVec.push_back(transMat);

        // --------------------  可视化  -----------------------
        cv::resize(labImg, labVis, cv::Size(labVis.cols, labVis.rows), 0, 0, cv::INTER_NEAREST);
        cv::imshow("range image", labVis);
        cv::waitKey(1);
//        labWriter << labImg;
        tag_fp.close();
        lab_fp.close();
        bin_fp.close();
    }
//    labWriter.release();
}

void writePointCloud2File(std::string dir, std::vector<std::string> &fileListVec)
{
    int wid = LINES_PER_BLK*BKNUM_PER_FRM;
    int len = PNTS_PER_LINE;
    printf("Saving labels to point cloud files.\n");
    for (int frm = 0; frm < fileListVec.size(); frm ++) {
        if (frm % 100 == 0) printf("%d\n", frm);
        std::ifstream tag_fp(dir+"/tag/"+fileListVec[frm]+".tag", std::ios::binary);
        std::ofstream lab_fp(dir+"/labels/"+fileListVec[frm]+".label", std::ios::binary);

        for (int i = 0; i < len; i ++) {
            for (int j = 0; j < wid; j ++) {
                // ------------------读入tag---------------------
                char t;
                tag_fp.read(&t, sizeof(char));
                if (t == 1) {
                    // ------------------写入label ----------------
                    int l = veloVec[frm].dat[i][j].lab;
                    lab_fp.write((char*)&l, sizeof(int));
                }
            }
        }
        tag_fp.close();
        lab_fp.close();
    }
    printf("Saved done.\n");
}
