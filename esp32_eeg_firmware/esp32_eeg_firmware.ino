/*
  EEG Acquisition Firmware — ESP32 + BioAmp EXG Pill
  ---------------------------------------------------
  Wiring:
    EXG Pill OUT  -> D34  (GPIO34 on most ESP32 DevKit boards)
    EXG Pill GND  -> GND
    EXG Pill VCC  -> 3V3  (do NOT use VIN/5V -- ESP32 ADC pins are 3.3V max)

  GPIO34 is an ADC1 pin and an input-only pin (no pull-up/pull-down, can't
  drive output) -- both are fine here since we only ever read from it.
  ADC1 pins stay reliable even with WiFi/BLE active, unlike ADC2 pins
  (e.g. GPIO2), which is why this was moved off D2.

  Output format:
    One integer ADC sample per line, printed at FS Hz, e.g.:
      512
      498
      531
      ...
    This matches what PYTHONCODE_fixed.py's parse_line() expects by default
    (plain int per line, no timestamp).

  Match these to your Python launch args:
    --port   -> whatever COM port this ESP32 enumerates as
    --baud   -> 115200 (matches Serial.begin below)
    --bits   -> 10 (matches analogReadResolution below)
*/

#define EXG_PIN 34         // D34 / GPIO34 -- ADC1, input-only
#define BAUD    115200
#define FS      250        // samples per second -- MUST match Python's FS = 250
#define ADC_BITS 8         // 8-bit: values 0-255 -- MUST match Python's --bits

const unsigned long SAMPLE_INTERVAL_US = 1000000UL / FS;
unsigned long lastSampleTime_us = 0;

void setup() {
  Serial.begin(BAUD);
  while (!Serial) { ; }  // wait for USB serial to be ready (harmless on most boards)

  analogReadResolution(ADC_BITS);   // 10-bit: values 0-1023
  analogSetAttenuation(ADC_11db);   // full ~0-3.3V input range

  // Note: no pinMode() call needed -- GPIO34 is input-only by hardware design,
  // and analogRead() handles its configuration internally.

  lastSampleTime_us = micros();
}

void loop() {
  unsigned long now_us = micros();

  // Fire at exactly FS Hz using a free-running interval, not delay(),
  // so timing stays accurate even if loop() does other work later.
  if (now_us - lastSampleTime_us >= SAMPLE_INTERVAL_US) {
    // advance by one interval (not "now") to avoid drift accumulating
    lastSampleTime_us += SAMPLE_INTERVAL_US;

    int sample = analogRead(EXG_PIN);
    Serial.println(sample);
  }
}
