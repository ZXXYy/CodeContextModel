#!/bin/bash

#path="/data0/xiaoyez/CodeContextModel/data/"
#projects="mylyn,Platform,PDE"
#
#python ./duplicate_handler.py --path "$path"  --projects "$projects"

path="/data0/xiaoyez/CodeContextModel/data/"

output="PDE_output.txt"
projects="PDE"
rm -rf "$output"
nohup python -u ./duplicate_handler.py --path "$path" --projects "$projects" >> "$output" 2>&1 &

output="Platform_output.txt"
projects="Platform"
rm -rf "$output"
nohup python -u ./duplicate_handler.py --path "$path" --projects "$projects" >> "$output" 2>&1 &

output="mylyn_output.txt"
projects="mylyn"
rm -rf "$output"
nohup python -u ./duplicate_handler.py --path "$path" --projects "$projects" >> "$output" 2>&1 &