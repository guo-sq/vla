#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 使能机械臂
import time
from piper_sdk import *

# 失能机械臂-ctrl_disable
def disable_piper(port):
    piper = C_PiperInterface_V2(port)
    piper.ConnectPort()
    while(piper.DisablePiper()):
        time.sleep(0.01)
    print("失能成功!!!!")

# 使能机械臂-ctrl_enable
def enable_piper(port):
    piper = C_PiperInterface_V2(port)
    print("---connect start---")
    piper.ConnectPort()
    print("---connect success---")
    time.sleep(0.1)
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    print("使能成功!!!!")

# 复位机械臂-ctrl_reset
# 将其设置为示教模式后，必须执行一次此操作。
def reset_piper(port):
    piper = C_PiperInterface_V2(port)
    piper.ConnectPort()
    piper.MotionCtrl_1(0x02,0,0)

def set_master_piper(port):
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    piper.MasterSlaveConfig(0xFA, 0, 0, 0)

# 设为关节运动？
def ctrl_piper_movJ(port):
    piper = C_PiperInterface_V2(port)
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    piper.GripperCtrl(0,1000,0x01, 0)
    factor = 57295.7795 #1000*180/3.1415926
    position = [0,0,0,0,0,0,0]
    count = 0
    while True:
        count  = count + 1
        # print(count)
        if(count == 0):
            print("1-----------")
            position = [0,0,0,0,0,0,0]
        elif(count == 300):
            print("2-----------")
            position = [0.2,0.2,-0.2,0.3,-0.2,0.5,0.08]
        elif(count == 600):
            print("1-----------")
            position = [0,0,0,0,0,0,0]
            count = 0
            break
        
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        joint_6 = round(position[6]*1000*1000)
        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        print(piper.GetArmStatus())
        print(position)
        time.sleep(0.001)
# v1204.1.txt
def set_can0(port="can0"):
    # ->设为主臂->reset->使能->使能->关节运动->失能
    print("set_master...")
    set_master_piper(port)
    print("reset_piper...")
    reset_piper(port)
    print("ctrl_piper_movJ...")
    # ctrl_piper_movJ(port)
    print("enable_piper..")
    # enable_piper(port)
    
    print("disable_piper...")
    disable_piper(port)

def set_can1(port="can1"):
    # ->关节运动->失能
    ctrl_piper_movJ(port)
    disable_piper(port)

if __name__ == "__main__":

    # set_can0("can0")
    # print("can0 finished.")
    # set_can1("can1")  
    # print("can1 finised.")
    enable_piper("can0")
    print("---0---")
    # enable_piper("can3")
    # print("---success---")
    # enable_piper("can2")
    # print("---2---")
    # enable_piper("can3")
    # print("---3---")
    



    