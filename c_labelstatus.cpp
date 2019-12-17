#include "c_labelstatus.h"
#include "header.h"

C_LabelStatus::C_LabelStatus()
{
    this->mode = 0;
    this->brushR = 10;
    this->curFrame = 0;
    this->isShowSingleFrame = 0;
    this->curInstanceLabel = 0;
    this->isFiltered = 1;
}
