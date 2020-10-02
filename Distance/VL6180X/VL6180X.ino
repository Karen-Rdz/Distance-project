/* This minimal example shows how to get single-shot range
measurements from the VL6180X.

The range readings are in units of mm. */

#include <Wire.h>
#include <VL6180X.h>

VL6180X sensor;

void setup() 
{
  Serial.begin(9600);
  Wire.begin(8);      // join i2c bus with address #8
  Wire.onRequest(requestEvent); // register event

  sensor.init();
  sensor.configureDefault();
  sensor.setTimeout(500);
}

void loop() 
{ 
  Serial.print(sensor.readRangeSingleMillimeters());
  if (sensor.timeoutOccurred()) { Serial.print(" TIMEOUT"); }

// function that executes whenever data is requested by master
// this function is registered as an event, see setup()
void requestEvent() {
  Wire.write("hello ");  // respond with message of 6 bytes
  // as expected by master
  
  Serial.println();
}
