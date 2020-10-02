#include "Adafruit_VL53L0X.h"

Adafruit_VL53L0X lox = Adafruit_VL53L0X();
uint8_t direccion = 0;
bool found = false;

void setup() {
  Serial.begin(115200);
    while (! Serial) {
    delay(1);
  }
    Serial.println("Adafruit VL53L0X test");
    while (direccion < 255 && !found){
      if (!lox.begin(direccion)) {
      Serial.print(F("Failed to boot VL53L0X  "));
      Serial.println (direccion);
      direccion++;
      delay (1);
  }else {
    Serial.println ("Si pude hacerlo jalar");
    found = true;
  }  
    }
  Serial.println ("Estoy vivo");                                  /*Para poder cambiar la dirección del sensor:
                                                                  Iniciar el sensor con la direción que tiene actual(begin(address)), 
                                                                  después indicarle la nueva dirección (set.Address(newAddress)).
                                                                  Al volver a hacerlo poner la nueva direccion al inicializarlo(begin(newAddress))*/
  if ( lox.setAddress(0x35)){
    Serial.println (" Empezo" ); 
  }else {
    Serial.println ("NO empezo");
  }
 
  // wait until serial port opens for native USB devices

  

  // power 
  Serial.println(F("VL53L0X API Simple Ranging example\n\n")); 
}


void loop() {
}
