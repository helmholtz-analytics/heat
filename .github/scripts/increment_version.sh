#!/bin/bash
version=$1 # assume version is passed as an argument
IFS='.' read -r -a parts <<< "$version" # split by dots
last_index=$(( ${#parts[@]} - 1 )) # get last index
parts[$last_index]=$(( ${parts[$last_index]} + 1 )) # increment last part
new_version=$(IFS=.; echo "${parts[*]}") # join by dots
echo $new_version # print new version
