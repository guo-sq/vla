import time
from piper_sdk import C_PiperInterface_V2

print("开始一分钟诊断...")
piper = C_PiperInterface_V2("can0")
piper.ConnectPort()
piper.EnablePiper()

error_frames = 0
zero_reads = 0
valid_reads = 0

for i in range(60):
    try:
        joint = piper.GetArmJointMsgs()
        gripper = piper.GetArmGripperMsgs()
        
        if joint.joint_state.joint_1 == 0:
            zero_reads += 1
        else:
            valid_reads += 1
            
        print(f"[{i:02d}] joint_1={joint.joint_state.joint_1}, zero_reads={zero_reads}, valid={valid_reads}")
        
    except Exception as e:
        error_frames += 1
        print(f"[{i:02d}] 错误: {e}")
    
    time.sleep(1)

print(f"\n=== 诊断结果 ===")
print(f"零值读取: {zero_reads}/60")
print(f"有效读取: {valid_reads}/60")
print(f"错误帧: {error_frames}/60")

if zero_reads > 50:
    print("🔴 机械臂固件异常，持续发送空数据")
elif error_frames > 10:
    print("🔴 CAN总线不稳定，物理层问题")
else:
    print("🟢 通信正常（临时故障）")