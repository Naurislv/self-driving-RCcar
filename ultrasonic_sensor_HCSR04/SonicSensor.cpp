#include <stdio.h>
#include <stdlib.h>
#include <wiringPi.h>

#define TRUE 1

#define TRIG 5
#define ECHO 6

void setup() {
	std::cout << "DDDD" << '\n';
    wiringPiSetup();
	std::cout << "CCCC" << '\n';
    pinMode(TRIG, OUTPUT);
    pinMode(ECHO, INPUT);

	std::cout << "BBBB" << '\n';
    //TRIG pin must start LOW
    digitalWrite(TRIG, LOW);

	std::cout << "AAAA" << '\n';
    delay(30);
}

int getCM() {
    //Send trig pulse
    digitalWrite(TRIG, HIGH);
    delayMicroseconds(20);
    digitalWrite(TRIG, LOW);

    //Wait for echo start
    while(digitalRead(ECHO) == LOW);

    //Wait for echo end
    long startTime = micros();
    while(digitalRead(ECHO) == HIGH);
    long travelTime = micros() - startTime;

    //Get distance in cm
    int distance = travelTime / 58;

    return distance;
}

int main(void) {
    setup();

    printf("Distance: %dcm\n", getCM());
    return 0;
}
