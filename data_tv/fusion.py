import numpy as np

import sys

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

if len(sys.argv) != 6:
    print(
        "Usage: python fusion.py <Result paths separeted by comma>"
        + " <Output directory> <Start topic ID> <End topic ID>"
        + " <Output an npy file of fusion result (0: not output, otherwise output)>"
    )
    sys.exit()

allResultDirs = sys.argv[1].split(",")
outputDir = sys.argv[2]
startTopicID = int(sys.argv[3])
endTopicID = int(sys.argv[4])
outputFusionResult = int(sys.argv[5])

print(">> All result directories: {}".format(allResultDirs))
print(">> Output directory: {}".format(outputDir))
print(
    ">> Start topic ID: {}, End topic ID: {}"
    .format(startTopicID, endTopicID)
)
print(">> Output npy of fusion result: {}".format(outputFusionResult))

shotInfos = []
with open("shotInfos.txt", "r") as f:
    for i, line in enumerate(f):
        line = line[:-1]
        tmp = line.split(" ")
        shotInfo = "{}_{}".format(tmp[0], int(tmp[1])+1)
        if i % 100000 == 0:
            print("{}th line: {} -> {}".format(i, line, shotInfo))
        shotInfos.append(shotInfo)

nbTopics = endTopicID - startTopicID + 1
simsFinal = np.zeros((nbTopics, len(shotInfos)))
# set_trace()

for resultDir in allResultDirs:
    
    resultFilename = resultDir + "sims_all.npy"
    print(">> Loading {}".format(resultFilename))
    sims = np.load(resultFilename)
    assert simsFinal.shape == sims.shape,\
        "Result shape mismatch ({} vs. {}"\
        .format(simsFinal.shape, sims.shape)
    """
    sims_mins = np.min(sims, 1)
    sims_maxs = np.max(sims, 1)
    sims_mins = sims_mins[:,np.newaxis]
    sims_maxs = sims_maxs[:,np.newaxis]
    sims = (sims - sims_mins) / (sims_maxs - sims_mins)
    """
    simsFinal += sims
    

for topicID in range(startTopicID, endTopicID+1):
    
    outputFilename = outputDir + str(topicID) + ".trec"
    print(
        ">> Output the fusion result for {}th topic into {}"
        .format(topicID, outputFilename)
    )
   
    # Sort shots based on their similarities to the topic_id-th topic
    ranks = np.argsort( simsFinal[topicID - startTopicID, :] )
    ranks = ranks[::-1]
    # set_trace()

    with open(outputFilename, 'w') as fout:
        for i in range(1000):
            if i == 0 or i== 999:
                print(
                    "--- {}th ranked shot: {} (sim:{})"
                    .format(i, shotInfos[ranks[i]], simsFinal[topicID-startTopicID,ranks[i]])
                )
            print(
                "{} 0 shot{} {} {} trec"
                .format(1000+topicID, shotInfos[ranks[i]], i+1, 9999-i),
                file=fout
            )
        
    # if topicID == 1:
    #     set_trace()


if outputFusionResult != 0:
    outputFusionResultFilename = outputDir + "sims_all.npy"
    print(
        ">> Output an npy file of fusion result into {}"
        .format(outputFusionResultFilename)
    )
    np.save(outputFusionResultFilename, simsFinal)
    # set_trace()

