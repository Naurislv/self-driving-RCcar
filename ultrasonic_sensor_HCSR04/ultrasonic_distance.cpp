/*
 *  hc-sr04.c:
 *	Simple test program to test the wiringPi functions
 */

#include <wiringPi.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int TRIG, ECHO;

static int ping()
{
	long ping      = 0;
	long pong      = 0;
	float distance = 0;
 	long timeout   = 500000; // 0.5 sec ~ 171 m

	pinMode(TRIG, OUTPUT);
	pinMode(ECHO, INPUT);

	// Ensure trigger is low.
	digitalWrite(TRIG, LOW);
	delay(50);

	// Trigger the ping.
	digitalWrite(TRIG, HIGH);
	delayMicroseconds(10);
	digitalWrite(TRIG, LOW);

	// Wait for ping response, or timeout.
	while (digitalRead(ECHO) == LOW && micros() < timeout) {
	}

	// Cancel on timeout.
	if (micros() > timeout) {
		printf("Out of range.\n");
		return 0;
	}

	ping = micros();

	// Wait for pong response, or timeout.
	while (digitalRead(ECHO) == HIGH && micros() < timeout) {
	}

	// Cancel on timeout.
	if (micros() > timeout) {
		printf("Out of range.\n");
		return 0;
	}

	pong = micros();

	// Convert ping duration to distance.
	distance = (float) (pong - ping) * 0.017150;
	printf("Distance: %.2f cm.\n", distance);

	return 1;
}

int main (int argc, char *argv[])
{
	if (argc != 3) {
		printf ("usage: %s <trigger> <echo>\n\nWhere:\n- trigger is the wiringPi trigger pin number.\n- echo is the wiringPi echo pin number.\nUsing trigger %d and echo %d.\n", argv[0], argv[1], argv[2]);
	} else {
		TRIG = atoi(argv[1]);
		ECHO = atoi(argv[2]);
	}

	printf ("Raspberry Pi wiringPi HC-SR04 Sonar test program.\n");

	if (wiringPiSetup () == -1) {
		exit(EXIT_FAILURE);
	}

	if (setuid(getuid()) < 0) {
		perror("Dropping privileges failed.\n");
		exit(EXIT_FAILURE);
	}

	ping();
	return 0;
}
