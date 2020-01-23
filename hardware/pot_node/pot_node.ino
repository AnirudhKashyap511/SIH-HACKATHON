/*
 * rosserial Publisher Example
 * Prints "hello world!"
 */

#include <ros.h>
#include <std_msgs/Float64.h>

ros::NodeHandle nh;

std_msgs::Float64 pot_value;
ros::Publisher chatter("chatter", &pot_value);

int ledPin = 13;
int ledPinRed = 12;
int ledPinBlue = 11;
void setup() {
  nh.initNode();
  pinMode(ledPinRed, OUTPUT);
  pinMode(ledPinBlue,OUTPUT);
  nh.advertise(chatter);
  int potPin = 2;    // select the input pin for the potentiometer
  
}

// the loop routine runs over and over again forever:
void loop() {
  int val;
  int sensorValue = analogRead(A0);
  val = map(sensorValue, 0, 1023, 0, 120);
  pot_value.data = val;
  if(val>50&&val<70)
    {
      digitalWrite(ledPinBlue,HIGH);
      delay(500);
      digitalWrite(ledPinBlue,LOW);
    }
    if(val>70)
    {
       digitalWrite(ledPinRed,HIGH);
      delay(500);
      digitalWrite(ledPinRed,LOW);
    }
  chatter.publish( &pot_value );
  nh.spinOnce();
  delay(1000);
}
