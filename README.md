# DRIVER-ALERT-SYSTEM

In order to ensure pedestrian safety and vehicle speed limits in areas with high pedestrian density/residential areas, a system to detect the traffic density/ school areas/ residential areas and accordingly warn the driver about the speed limit for vehicles. The participating teams may use vision based systems which sense the speed limit signs or sign boards for schools/hospitals/accident prone areas etc. and may decide upon the safe speed value with which the vehicle need to operate. The system needs to provide a warning to the driver, if the speed limits are breached. For the purpose of demonstration, the warning may be provided using a LCD or a seven segment simple display and a buzzer. The sign boards for the purpose of demonstration needs to be made according to IRC standards and the same can be used by the system for demonstration purposes.

This idea is proposed by ARAI organization. We chose to provide a solution using vision based system.

 - Hardware stack
 1. Simulate a speedometer using potentiometer where resistance is mapped to speed.
 2. We used ROS and Arduino to integrate hardware and software.
 3. Speech based warning is triggered when driver breaches the speed limit.

