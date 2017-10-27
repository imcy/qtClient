#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "pymodel.h"
#include <QDebug>

Thread threadA; //必须放在前面
pyModel pymodel;

int cli_sockfd;/*客户端SOCKET */
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    initSocket();
    connect(&threadA, SIGNAL(sendData(QString)), this, SLOT(receiveData(QString))); //关联发送和接受函数

}
void MainWindow::initSocket(){

    /*设置初始化IP地址*/

    QString valueStr="125.216.243.10";
    ui->ipEdit->setText(valueStr);

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_click_clicked()
{
    char *mm="leave";
    send(cli_sockfd,mm,sizeof(mm),0);
    ::close(cli_sockfd);
    if(threadA.isRunning()){
        threadA.terminate();
        threadA.wait();
    }
    exit(0);
}

void MainWindow::on_connectRemote_clicked()
{

    QString valueStr=ui->ipEdit->text();
    ui->connectRemote->setEnabled(false);
    struct sockaddr_in ser_addr;/* 服务器的地址*/
    cli_sockfd=socket(AF_INET,SOCK_STREAM,0);//创建连接的SOCKET
    if(cli_sockfd<0){
        //创建失败
        ui->textOutput->append("创建socket失败");
        exit(0);
    }
    ui->textOutput->append("创建socket成功");
    // 初始化服务器地址
    socklen_t addrlen=sizeof(struct sockaddr_in);
    bzero(&ser_addr,addrlen);
    ser_addr.sin_family=AF_INET;
    ser_addr.sin_addr.s_addr = inet_addr("125.216.243.10");  ///服务器ip
    ser_addr.sin_port=htons(SERVER_PORT);
    if(::connect(cli_sockfd,(struct sockaddr*)&ser_addr,sizeof(ser_addr))!=0)//请求连接
    {
        //连接失败
        ui->textOutput->append("连接失败");
        ::close(cli_sockfd);
        ui->connectRemote->setEnabled(true);
    }else{
        ui->textOutput->append("已连接到服务器"+valueStr);
        threadA.start();
    }

}

void MainWindow::on_send_clicked()
{
    QString valueStr=ui->inputEdit->toPlainText();
    char msg[MAX_MSG_SIZE];/* 缓冲区*/
    QByteArray ba = valueStr.toLatin1();
    char *mm = ba.data();
    strcpy(msg,mm);
    if(send(cli_sockfd,msg,sizeof(msg),0)==-1){
        /*发送数据*/
        ui->textOutput->append("发送失败");
    }else{
         memset(msg,0,MAX_MSG_SIZE);
         ui->textOutput->append("client:"+valueStr);
         ui->inputEdit->setPlainText("");
    }

}
void MainWindow::receiveData(QString data){
    ui->textOutput->append("server:"+data);
}


void MainWindow::on_initPython_clicked()
{
    pymodel.initModel();
}

void MainWindow::on_readtxt_clicked()
{
    QFile f("cut.txt");
    if(!f.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Open failed." ;
    }
    QTextStream txtInput(&f);
    QString lineStr;
    while(!txtInput.atEnd())
    {
        lineStr = txtInput.readLine();
        ui->textOutput->append(lineStr);
    }

    f.close();
}
