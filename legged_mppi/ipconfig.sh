#!/bin/bash

sudo ifconfig lo multicast
sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo

sudo ifconfig enp47s0 down
sudo ifconfig enp47s0 up 192.168.123.161 netmask 255.255.255.0

