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
              emit sendData("准备收图片\n");
          }else{
              emit sendData(QString(msg));
              memset(msg,0,MAX_MSG_SIZE);
          }
      }
    }
}

void Thread::stop()
{
    stopped = true;
}

void Thread::setMessage(QString message)
{
    messageStr = message;
}
