hw_timer_t *timer = NULL;
volatile bool sampleReady = false;
volatile uint16_t latestSample = 0;
void IRAM_ATTR onTimer() {
  uint32_t sum = 0;
  for (int i = 0; i < 8; i++) sum += analogRead(34);
  latestSample = sum / 8;
  sampleReady = true;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);      // 0-4095 across 0-3.3V

  timer = timerBegin(1000000);           // 1 MHz tick
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 4000, true, 0);      // 4000us = 250 Hz
}

void loop() {
  if (sampleReady) {
    sampleReady = false;
    Serial.print("TS:");
    Serial.print(millis());
    Serial.print(",VAL:");
    Serial.println(latestSample);
  }
}