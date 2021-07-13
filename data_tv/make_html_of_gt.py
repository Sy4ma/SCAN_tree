
import sys
import os
import shutil


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


if len(sys.argv) != 5:
    print(
        "Usage: python edit_for_submission.py <Ground truth filename> <Output directory>"
        + " <Start topic ID> <End topic ID>"
    )
    sys.exit()

gtFilename = sys.argv[1]
print(">> Ground truth filename: {}".format(gtFilename))
outputDir = sys.argv[2]
startTopicID = int(sys.argv[3])
endTopicID = int(sys.argv[4])

correctShots = { tid: [] for tid in range(startTopicID, endTopicID+1) }

with open(gtFilename) as fgt:
    for i, line in enumerate(fgt):
        line = line[:-1]
        line_elems = [e for e in line.split(" ") if len(e) > 0]
        if int(line_elems[-1]) == 1:
            topicID = int(line_elems[0]) - 1000
            print(
                "Topic {}: {}th correct shot {}"
                .format(topicID, len(correctShots[topicID]), line_elems[2])
            )
            correctShots[topicID].append(line_elems[2])
            # if len(correctShots[topicID]) >= 10:
            #     set_trace()

# set_trace()

keyframeRootDir = "/home/kimiaki/TRECVID2021/keyframe/"

for topicID in range(startTopicID, endTopicID+1):
    
    htmlDir = outputDir + str(topicID) + "/"
    htmlFilename = htmlDir + "ground_truth.html"
    htmlImageDir = htmlDir + "images/"
    
    print("##### Processing {}th topic #####".format(topicID))
    print(">> Directory to store an html file: {}".format(htmlDir))
    print(">> Directory to store iamges used in the above html file: {}".format(htmlImageDir))
    os.mkdir(htmlDir)
    os.mkdir(htmlImageDir)
    
    print(">> # of correct shots = {}".format( len(correctShots[topicID]) ))
    
    print(">> Open the output html file, {}".format(htmlFilename))
    fout = open(htmlFilename, "w")
    
    print("<html><head><title>{}</title></head>\n<body>\n".format(htmlFilename), file=fout)
    
    for i, correctShot in enumerate( correctShots[topicID] ):
        
        # if i >= 100:
        #     break
        
        if i % 10 == 0:
            print("<table cellpadding=5><tr>\n", file=fout)
        
        videoID = (correctShot.split("_"))[0][4:]
        keyframeFilename = keyframeRootDir + videoID + "/" + correctShot + "_RKF.png"
        keyframeFilename2 = htmlImageDir + correctShot + "_RKF.png"
        if i % 50 == 0:
            print(
                "-- Copy the {}th ground truth from {} to {}"
                .format(i, keyframeFilename, keyframeFilename2)
            )
        shutil.copyfile(keyframeFilename, keyframeFilename2)
        
        keyframeFilenameInHTML = "./images/" + correctShot + "_RKF.png"
        print(
            "<td><table align=middle><tr align=middle><td><img src=\"{}\" width=100></td></tr><tr align=middle><td>"
            .format(keyframeFilenameInHTML), file=fout
        )
        print(
            "<font size=1>{}:{}</font></td></tr></table></td>\n"
            .format(i, correctShot), file=fout
        )
        
        if i % 10 == 9:
            print("</tr></table>\n \n", file=fout)
        
    
    print("</body></html>", file=fout)
    
    fout.close()
    # set_trace()
    

