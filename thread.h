#ifndef THREAD_H
#define THREAD_H
#include <QThread>
#include<QProcess>
#include "mainwindow.h"

typedef unsigned long DWORD;

class Thread:public QThread
{
     Q_OBJECT
public:
    Thread();
    void setMessage(QString message);
    void stop();
    void receiveImage();
    void sendres();
protected:
    void run();
private:
    QString messageStr;
    volatile bool stopped;
    void printMessage();
signals:
    void sendData(QString);   //用来传递数据的信号

};

#endif // THREAD_H
