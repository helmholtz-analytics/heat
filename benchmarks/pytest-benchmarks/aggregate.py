import io
import json
import os

if os.path.isfile('aggregate.json') and os.access('aggregate.json', os.R_OK):
    # checks if file exists
    inFile = open('output.json', "r")
    new_data = json.load(inFile)
    del new_data["benchmarks"][0]["stats"]["data"]
    new_data["benchmarks"][0]["commit_info"] = new_data["commit_info"]
    with open('aggregate.json', 'r+') as outFile:
        file_data =json.load(outFile)
        file_data.append(new_data["benchmarks"][0])
        outFile.seek(0)
        json.dump(file_data,outFile,indent=4)
    inFile.close()
    outFile.close()

else:
    print ("aggregate file is missing or is not readable, creating file...")
    inFile = open('output.json', "r")
    new_data = json.load(inFile)
    del new_data["benchmarks"][0]["stats"]["data"]
    new_data["benchmarks"][0]["commit_info"] = new_data["commit_info"]
    with open('aggregate.json', 'w') as outFile:
        outFile.write(json.dumps([new_data["benchmarks"][0]]))
    inFile.close()
    outFile.close()

