/* This example shows how to get single-shot range
 measurements from the VL53L0X. The sensor can optionally be
 configured with different ranging profiles, as described in
 the VL53L0X API user manual, to get better performance for
 a certain application. This code is based on the four
 "SingleRanging" examples in the VL53L0X API.

 The range readings are in units of mm. */

#include <Wire.h>
#include <VL53L0X.h>

VL53L0X sensor2;
VL53L0X sensor3;


bool InicializarSensor (uint8_t address, VL53L0X *sensor, uint8_t pin ){
  Serial.println ("Antes del if");
  if (sensor->init(true)){
     sensor->setAddress(address);
     Serial.println ("Dentro del if");
     return true;
  }else{
     return false;
  }
}
// Uncomment this line to use long range mode. This
// increases the sensitivity of the sensor and extends its
// potential range, but increases the likelihood of getting
// an inaccurate reading because of reflections from objects
// other than the intended target. It works best in dark
// conditions.

//#define LONG_RANGE


// Uncomment ONE of these two lines to get
// - higher speed at the cost of lower accuracy OR
// - higher accuracy at the cost of lower speed

//#define HIGH_SPEED
#define HIGH_ACCURACY


void setup()
{
  Serial.begin(9600);
  Wire.begin();
  pinMode (2, OUTPUT); 
  pinMode (3, OUTPUT);
  digitalWrite (2, LOW);
  digitalWrite (3, LOW);
    delay (10);
  digitalWrite (2, HIGH);
  digitalWrite (3, HIGH);
  digitalWrite (3, LOW);
   Serial.println ("Antes de iniciar");
  InicializarSensor(0x35, &sensor2, 2);
  digitalWrite (3, HIGH);
  InicializarSensor(0x36, &sensor3, 3); 
    Serial.println ("Despues de iniciar");
  sensor2.setTimeout(500);
  sensor3.setTimeout(500);

#if defined LONG_RANGE
  // lower the return signal rate limit (default is 0.25 MCPS)
  sensor2.setSignalRateLimit(0.1);
  // increase laser pulse periods (defaults are 14 and 10 PCLKs)
  sensor2.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
  sensor2.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
#endif

#if defined HIGH_SPEED
  // reduce timing budget to 20 ms (default is about 33 ms)
  sensor2.setMeasurementTimingBudget(20000);
  sensor3.setMeasurementTimingBudget(20000);
#elif defined HIGH_ACCURACY
  // increase timing budget to 200 ms
  sensor2.setMeasurementTimingBudget(200000);
  sensor3.setMeasurementTimingBudget(200000);
#endif
}

void loop()
{
  Serial.print(sensor2.readRangeSingleMillimeters());
  Serial.print("   ");
  Serial.println (sensor3.readRangeSingleMillimeters());
  if (sensor2.timeoutOccurred()) { Serial.print(" TIMEOUT"); }

  Serial.println();
}
