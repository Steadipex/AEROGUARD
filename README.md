# AEROGUARD

### Environment-Adaptive Multi-Sensor Anti-Drone Detection and Tracking System

AEROGUARD is a prototype anti-drone detection and tracking system designed to improve target tracking reliability under harsh environmental conditions such as low temperature, reduced atmospheric pressure and increased vibration.

## Problem

Conventional camera-based drone detection can become unreliable under adverse environmental and operating conditions.

AEROGUARD addresses this challenge using a multi-sensor approach combining:

- Camera-based AI detection
- RF-based detection
- Radar-based detection
- Target tracking
- Environmental monitoring
- System health estimation
- Adaptive sensor fusion

## System Architecture

```text
              DRONE
                |
        +-------+-------+
        |               |
     CAMERA          RF / RADAR
        |               |
        v               v
   AI Detection    Sensor Confidence
        |               |
        +-------+-------+
                |
                v
         SENSOR FUSION
                |
                v
        TARGET TRACKING
                |
                v
       SYSTEM HEALTH MODEL
                |
                v
       ADAPTIVE OPERATION
