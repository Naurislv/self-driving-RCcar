"""
https://github.com/alaudet/hcsr04sensor

pip install -U hcsr04sensor
"""

from hcsr04sensor import sensor


def main():
    '''Calculate the distance of an object in centimeters using a HCSR04 sensor
       and a Raspberry Pi
     '''

    trig_pin = 24
    echo_pin = 23
    value = sensor.Measurement(trig_pin, echo_pin)

    while (1):
        raw_measurement = value.raw_distance(sample_size=1, sample_wait=0.001)
        metric_distance = value.distance_metric(raw_measurement)
        if (metric_distance <= 200):
            print("The Distance = {} centimeters".format(metric_distance))


if __name__ == "__main__":
    main()
