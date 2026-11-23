# CONFERENCE-ON-COMPUTER-SCIENCE-PROJECT
ขั้นตอนการใช้งานสำหรับเฟรม
1. เปิดระบบโดยเชื่อมต่อ TCP IP ด้วยคำสั่งเหล่านี้
   1.1 nmcli device status
   1.2 sudo nmcli con add type ethernet ifname enp3s0 con-name dobot \ ipv4.method manual ipv4.addresses 192.168.1.10/24 ipv4.never-default yes
      sudo nmcli con up dobot
   1.3 ping -c 3 192.168.1.6
       python3 Project.py --dobot-test 192.168.1.6
   1.4 ip -br addr show enxc84d4429794c
   1.5 sudo nmcli con modify dobot connection.interface-name enxc84d4429794c
       sudo nmcli con up dobot
2. การทำ Hand eye calibration
   2.1 ros2 launch realsense2_camera rs_launch.py initial_reset:=true align_depth.enable:=true
   2.2 python3 Handeye_calibration.py --ros-args -p dobot_ip:=192.168.1.6
      2.2.1 กด p เพื่อจำพิกัดหุ่นยนต์
      2.2.2 กด space bar เพื่อแสกนวัตถุสีเหลืองเพื่อเทียบกับตำแหน่งหุ่นยนต์
      2.2.3 ทำ ไปเรื่อยๆ จนกว่าจะครบ 30 จุด ตำแหน่งเรากำหนดให้แล้วให้เปลี่ยนตามนี้
3. การเก็บผลการทดลอง
    ~/.local/bin/dobot-planner
      
   
   
