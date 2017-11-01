#include "thread.h"
#include <QDebug>

extern int cli_sockfd;
Thread::Thread()
{
    stopped = false;
}
void Thread::run()
{
    int res;
    char msg[MAX_MSG_SIZE];/* 缓冲区*/
    while (1) {
      if((res=recv(cli_sockfd,msg,MAX_MSG_SIZE,0))==-1) //接受数据
      {
            emit sendData("失去连接\n");
            return;
      }else{

          if(strcmp(msg,"sendImage")==0){
              emit sendData("准备收图片");
              receiveImage();
          }else{
              emit sendData(QString(msg));
              memset(msg,0,MAX_MSG_SIZE);
          }
      }
    }
}
void Thread::receiveImage(){
    char buffer[MAX_MSG_SIZE];/* 缓冲区*/
    FILE *stream;
    unsigned long long file_size=0;
    int res;
    if((res=recv(cli_sockfd,(char *)&file_size,sizeof(unsigned long long)+1,0))==-1) //接受数据
    {
          emit sendData("失去连接");
          return;
    }else{
       // unsigned short maxvalue=file_size;
       qDebug()<<file_size;
       stream = fopen("../images/cut.png","w");
       DWORD dwNumberofBytesRecv=0;
       DWORD dwCountofBytesRecv=0;
       memset(buffer,0,MAX_MSG_SIZE);
       do{
           dwNumberofBytesRecv=recv(cli_sockfd,buffer,sizeof(buffer),0);
           fwrite(buffer,sizeof(char),dwNumberofBytesRecv,stream); //按字节写入图片
           dwCountofBytesRecv+=dwNumberofBytesRecv;
       }while(file_size-dwCountofBytesRecv);
       emit sendData("文件接收成功");
       fclose(stream);
       QProcess::execute("python democy.py");
       sendres(); //发送计算结果
    }


}
void Thread::sendres()
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
        emit sendData(lineStr);

        char msg[MAX_MSG_SIZE];/* 缓冲区*/
        QString flag="sendResult";
        QByteArray ba = flag.toLatin1();
        char *mm = ba.data();
        strcpy(msg,mm);
        //发送结果标志位
        if(send(cli_sockfd,msg,sizeof(msg),0)==-1){
            /*发送数据*/
            emit sendData("发送失败");
        }else{
            memset(msg,0,MAX_MSG_SIZE);
            emit sendData("准备发送计算结果");

            //发送结果
            ba = lineStr.toLatin1();
            mm = ba.data();
            strcpy(msg,mm);
            if(send(cli_sockfd,msg,sizeof(msg),0)==-1){
                /*发送数据*/
                emit sendData("发送失败");
            }else{
                memset(msg,0,MAX_MSG_SIZE);
                emit sendData("已经发送计算结果");
            }
        }
    }
    f.close();

}

void Thread::stop()
{
    stopped = true;
}

void Thread::setMessage(QString message)
{
    messageStr = message;
}
