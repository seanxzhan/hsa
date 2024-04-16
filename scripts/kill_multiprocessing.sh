#!/bin/bash

name=$1

kill -9 `ps -ef | grep ${name} | grep -v grep | awk '{print $2}'`