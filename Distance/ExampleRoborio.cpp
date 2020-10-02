I2C *i2c = new I2C(I2C::Port::kOnboard, 2);
 uint8_t lightPattern[1];
 lightPattern[0] = 1; // Probably better to define enums for various light modes, but set a light mode here
 uint8_t arduinoData[1];
 i2c->Transaction(lightPattern, 1, arduinoData, 1);