#include "pymodel.h"
#include <QDebug>
#include<QProcess>

pyModel::pyModel()
{

}
void pyModel::initModel(){

    QProcess::execute("python democy.py");
}
