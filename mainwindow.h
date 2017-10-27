#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/shm.h>
#include "thread.h"
#include <QFile>
#define SERVER_PORT 5150
#define MAX_MSG_SIZE 1024

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
private slots:
    void on_click_clicked();
    void initSocket();
    void on_connectRemote_clicked();
    void on_send_clicked();
    void receiveData(QString data);   //接收传递过来的数据的槽
    void on_initPython_clicked();

    void on_readtxt_clicked();

private:
    Ui::MainWindow *ui;


};
#endif // MAINWINDOW_H
