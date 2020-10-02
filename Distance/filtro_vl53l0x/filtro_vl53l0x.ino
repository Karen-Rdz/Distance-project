#include "Adafruit_VL53L0X.h"

Adafruit_VL53L0X sensor2;
Adafruit_VL53L0X sensor3;

#define  SHUT_2 2
#define SHUT_3 3

bool Inicializar_sensor(uint8_t pin, Adafruit_VL53L0X &sensor, uint8_t direccion){
 Serial.println (" Algo "); 
    if (!sensor.begin(direccion)) {   
      Serial.print(F("Failed to boot VL53L0X_"));
      Serial.println(direccion);
      return false;
  }
   Serial.print(" Sensor jalo");
   Serial.println (direccion); 
   return true;
}


class Filter
{
public:
  Filter(double* source, double errorMeasurement,double estUncertainty,double variance);
  double kalmanFilter();
  ~Filter();

private:
  double* source;
  double previousEst;
  double errorMeasurement;
  double estUncertainty;
  double variance;
  double errEstimate;
};
Filter::Filter(double* source, double errorMeasurement, double estUncertainty, double variance){
  this->errorMeasurement = errorMeasurement;
  this->estUncertainty = estUncertainty;
  this->variance = variance;
  this->source = source;
  previousEst = *source;
  errEstimate = estUncertainty;
}

double Filter::kalmanFilter() {

  double kGain = errEstimate / (errEstimate + errorMeasurement);
  double estimate = previousEst + kGain * (*source - previousEst);
  errEstimate = (1.0 - kGain) * errEstimate + fabs(previousEst - estimate)*variance;
  previousEst = estimate;
  return estimate;
}
Filter::~Filter()
{
}

  double medida;
  double medida1; 
  
  double resultado;
  double resultado1;
  
  Filter filtro(&medida, 5, 5,4.789);
  Filter filtro1(&medida1, 5,5, 4.789);



void setup() {
  Serial.begin(115200);
  Serial.println ("Inicio");
 pinMode(SHUT_2, OUTPUT);
 pinMode(SHUT_3, OUTPUT);
 digitalWrite(SHUT_2, LOW);
 digitalWrite(SHUT_3, LOW);
   delay(10);
 digitalWrite(SHUT_2, HIGH);
 digitalWrite(SHUT_3, HIGH);
  while (! Serial) {
    delay(1);
  }   
digitalWrite(SHUT_3, LOW); 
    if (!sensor2.begin(0x35)){
      while (1);
    }
   digitalWrite(SHUT_3, HIGH);
    if (!sensor3.begin(0x36)){ 
      while(1); 
    }
  }
void loop() {
  VL53L0X_RangingMeasurementData_t measure;
  VL53L0X_RangingMeasurementData_t measure1;
 
  medida = measure.RangeMilliMeter;
  medida1 = measure1.RangeMilliMeter;
  
  resultado = filtro.kalmanFilter();
  resultado1 = filtro1.kalmanFilter();
    
  Serial.print("Reading a measurement... ");
  sensor2.rangingTest(&measure, false); // pass in 'true' to get debug data printout!
  sensor3.rangingTest(&measure1, false);
  
  if (measure.RangeStatus != 4 and measure1.RangeStatus != 4) {  // phase failures have incorrect data
    //Serial.print("Distance (mm): "); 
    Serial.println(medida);Serial.print ("  ");Serial.println (medida1);
  } else {
    //Serial.println(" out of range ");
  }
    
  delay(100);
}
