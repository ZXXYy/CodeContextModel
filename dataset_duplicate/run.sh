#!/bin/bash

#path="/data0/xiaoyez/CodeContextModel/data/"
#output="output.txt"
#projects="mylyn,Platform,PDE"
#
#python ./duplicate_handler.py --path "$path" --output "$output" --projects "$projects"

path="/data0/xiaoyez/CodeContextModel/data/"

output="PDE_output.txt"
projects="PDE"
python ./duplicate_handler.py --path "$path" --output "$output" --projects "$projects"

output="Platform_output.txt"
projects="Platform"
python ./duplicate_handler.py --path "$path" --output "$output" --projects "$projects"

output="mylyn_output.txt"
projects="mylyn"
python ./duplicate_handler.py --path "$path" --output "$output" --projects "$projects"