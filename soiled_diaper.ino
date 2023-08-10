/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <Soiled-Diaper_inferencing.h>
#include "Arduino_BHY2.h"
#include <ArduinoBLE.h>
#include "Nicla_System.h"

/** Struct to link sensor axis name to sensor value function */
typedef struct{
    const char *name;
    float (*get_value)(void);

}eiSensors;

/** Number sensor axes used */
#define NICLA_N_SENSORS     2


/* Private variables ------------------------------------------------------- */
static const bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

Sensor gas(SENSOR_ID_GAS);
SensorBSEC bsec(SENSOR_ID_BSEC);

BLEService dataService("07e1f0e2-b68f-4f71-8bfc-19b3b0427b68"); 
BLEStringCharacteristic inferenceCharacteristic("5d75d723-fce9-471b-93e2-6c625ef6d634", BLERead | BLENotify, 20);
BLEStringCharacteristic iaqCharacteristic("5d75d724-fce9-471b-93e2-6c625ef6d634", BLERead | BLENotify, 20);

static bool ei_connect_fusion_list(const char *input_list);

//I normalized in data collection so need to do that here as well
//gas / 1000
static float get_gas(void){return (gas.value()/1000.00);}
static float get_iaq(void){return bsec.iaq();}

static int8_t fusion_sensors[NICLA_N_SENSORS];
static int fusion_ix = 0;

/** Used sensors value function connected to label name */
eiSensors nicla_sensors[] =
{
    //"hum", &get_humidity,
    "gas", &get_gas,
    "iaq",&get_iaq,
};

/**
* @brief      Arduino setup function
*/
void setup()
{
    /* Init serial */
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    //while (!Serial);
    Serial.println("Soiled Diaper Detector\r\n");

    /* Connect used sensors */
    if(ei_connect_fusion_list(EI_CLASSIFIER_FUSION_AXES_STRING) == false) {
        ei_printf("ERR: Errors in sensor list detected\r\n");
        return;
    }
    nicla::begin();
    nicla::leds.begin();
    nicla::leds.setColor(blue);
    /* Init & start sensors */
    BHY2.begin(NICLA_I2C);
    //hum.begin();
    gas.begin();
    bsec.begin();

    //Start BLE service
    if (!BLE.begin()) {
      Serial.println("starting BLE failed!");
      while (1);
    }
    // set the local name peripheral advertises
    BLE.setLocalName("DiaperDemo");
    // set the UUID for the service this peripheral advertises:
    BLE.setAdvertisedService(dataService); 
    // add the characteristics to the service
    dataService.addCharacteristic(inferenceCharacteristic);
    dataService.addCharacteristic(iaqCharacteristic);  
    inferenceCharacteristic.writeValue("");
    iaqCharacteristic.writeValue("");   
    // add the service
    BLE.addService(dataService);
    // start advertising
    //BLE.setAdvertisingInterval(80);
    BLE.advertise();
    Serial.println("Bluetooth device active, waiting for connections...");  
}

/**
* @brief      Get data and run inferencing
*/
void loop()
{
    BLEDevice central = BLE.central();
    // if a central is connected to the peripheral:
    if (central.discoverAttributes()) {
      // turn on the LED to indicate the connection:
      nicla::leds.setColor(green);
      Serial.print("Connected to central: ");
      Serial.println(central.address());
    }
    else {
      nicla::leds.setColor(red);
    }
    ei_printf("\nStarting inferencing in 2 seconds...\r\n");

    delay(2000);

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != fusion_ix) {
        ei_printf("ERR: Nicla sensors don't match the sensors required in the model\r\n"
        "Following sensors are required: %s\r\n", EI_CLASSIFIER_FUSION_AXES_STRING);
        return;
    }

    ei_printf("Sampling...\r\n");

    // Allocate a buffer here for the values we'll read from the IMU
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
        // Determine the next tick (and then sleep later)
        int64_t next_tick = (int64_t)micros() + ((int64_t)EI_CLASSIFIER_INTERVAL_MS * 1000);

        // Update function should be continuously polled
        BHY2.update();

        for(int i = 0; i < fusion_ix; i++) {
            buffer[ix + i] = (float)nicla_sensors[fusion_sensors[i]].get_value();
        }

        int64_t wait_time = next_tick - (int64_t)micros();

        if(wait_time > 0) {
            delayMicroseconds(wait_time);
        }
    }

    // Turn the raw buffer in a signal which we can the classify
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("ERR:(%d)\r\n", err);
        return;
    }

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR:(%d)\r\n", err);
        return;
    }
    int iaqVal = get_iaq();
    String iaqString = String(iaqVal);
    String infString = "";
    // print the predictions
    //store the location of the highest classified label
    int maxLabel = 0;
    float maxValue = 0.0;
    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.):\r\n",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if(result.classification[ix].value > maxValue) {
          maxLabel = ix;
          maxValue = result.classification[ix].value;
        }
        ei_printf("%s: %.5f\r\n", result.classification[ix].label, result.classification[ix].value);
    }
    infString = String(maxLabel);
    //send iaq and inference results to BLE
    if(central.connected()) {
      iaqCharacteristic.writeValue(iaqString);
      //if the gesture indicates a checkpoint movement, send notification to app
      if(infString == "soiled")
      {
        inferenceCharacteristic.writeValue("1");
      }
    }
    else {
      nicla::leds.setColor(red);
    }   

}

#if !defined(EI_CLASSIFIER_SENSOR) || (EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_FUSION && EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER)
#error "Invalid model for current sensor"
#endif


/**
 * @brief Go through nicla sensor list to find matching axis name
 *
 * @param axis_name
 * @return int8_t index in nicla sensor list, -1 if axis name is not found
 */
static int8_t ei_find_axis(char *axis_name)
{
    int ix;
    for(ix = 0; ix < NICLA_N_SENSORS; ix++) {
        if(strstr(axis_name, nicla_sensors[ix].name)) {
            return ix;
        }
    }
    return -1;
}

/**
 * @brief Check if requested input list is valid sensor fusion, create sensor buffer
 *
 * @param[in]  input_list      Axes list to sample (ie. "accX + gyrY + magZ")
 * @retval  false if invalid sensor_list
 */
static bool ei_connect_fusion_list(const char *input_list)
{
    char *buff;
    bool is_fusion = false;

    /* Copy const string in heap mem */
    char *input_string = (char *)ei_malloc(strlen(input_list) + 1);
    if (input_string == NULL) {
        return false;
    }
    memset(input_string, 0, strlen(input_list) + 1);
    strncpy(input_string, input_list, strlen(input_list));

    /* Clear fusion sensor list */
    memset(fusion_sensors, 0, NICLA_N_SENSORS);
    fusion_ix = 0;

    buff = strtok(input_string, "+");

    while (buff != NULL) { /* Run through buffer */
        int8_t found_axis = 0;

        is_fusion = false;
        found_axis = ei_find_axis(buff);

        if(found_axis >= 0) {
            if(fusion_ix < NICLA_N_SENSORS) {
                fusion_sensors[fusion_ix++] = found_axis;
            }
            is_fusion = true;
        }

        buff = strtok(NULL, "+ ");
    }

    ei_free(input_string);

    return is_fusion;
}
