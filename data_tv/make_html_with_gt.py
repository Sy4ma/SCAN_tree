
import sys
import os
import shutil


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


if len(sys.argv) != 3:
    print(
        "Usage: python edit_for_submission.py <Traget .trec file> <Ground truth filename)\n"
        + "NOTE1: The resulting html file will be save in the same directory to .trec file.\n"
        + "NOTE2: Regarding the target .trec file's filanme, the topic ID proceeds the extension .trec"
    )
    sys.exit()

targetFilename = sys.argv[1]
gtFilename = sys.argv[2]
topicID = int( (targetFilename[:targetFilename.rfind(".trec")])[-3:] )
htmlDir = targetFilename[:targetFilename.rfind(".trec")] + "/"
htmlImageDir = htmlDir + "images/"

print(">> Target .trec filename: {}".format(targetFilename))
print(">> Topic ID: {}".format(topicID))
print(">> Ground truth filename: {}".format(gtFilename))
print(">> Directory to store an html file: {}".format(htmlDir))
print(">> Directory to store iamges used in the above html file: {}".format(htmlImageDir))

correctShots = {}
with open(gtFilename) as fgt:
    for i, line in enumerate(fgt):
        line = line[:-1]
        line_elems = [e for e in line.split(" ") if len(e) > 0]
        if int(line_elems[0]) - 1000 == topicID \
            and int(line_elems[-1]) == 1:
            # print("{}th line: {}".format(i, line_elems))
            correctShots[line_elems[2]] = 1
print(">> # of correct shots = {}".format(len(correctShots)))

os.mkdir(htmlDir)
os.mkdir(htmlImageDir)


keyframeRootDir = "/home/kimiaki/TRECVID2021/keyframe/"
htmlFilename = htmlDir + "result.html"
print(">> Open the output html file, {}".format(htmlFilename))
fout = open(htmlFilename, "w")

print("<html><head><title>{}</title></head>\n<body>\n".format(htmlFilename), file=fout)

print(">> Open the target file, {}".format(targetFilename))
with open(targetFilename, "r") as fin:
    
    nbCorrectShotsRetrieved = 0
    for i, line in enumerate(fin):
        
        if i >= 100:
            break

        if i % 10 == 0:
            print("<table cellpadding=5><tr>\n", file=fout)

        tmp = line[:-1].split(" ")
        if i % 50 == 0:
            print("{} - {}".format(i+1, tmp[2]))
        videoID = (tmp[2].split("_"))[0][4:]
        
        keyframeFilename = keyframeRootDir + videoID + "/" + tmp[2] + "_RKF.png"
        keyframeFilename2 = htmlImageDir + tmp[2] + "_RKF.png"
        if i % 50 == 0:
            print(
                "-- Copy the {}th image from {} to {}"
                .format(i, keyframeFilename, keyframeFilename2)
            )
        shutil.copyfile(keyframeFilename, keyframeFilename2)
        
        keyframeFilenameInHTML = "./images/" + tmp[2] + "_RKF.png"
        if tmp[2] in correctShots:
            nbCorrectShotsRetrieved += 1
            print(
                "++ {}th ranked shot {} is correct ({})"
                .format(i, tmp[2], nbCorrectShotsRetrieved)
            )
            print(
                "<td><table align=middle bgcolor=\"red\"><tr align=middle><td><img src=\"{}\" width=100></td></tr><tr align=middle><td>"
                .format(keyframeFilenameInHTML), file=fout
            )
        else:
            print(
                "<td><table align=middle><tr align=middle><td><img src=\"{}\" width=100></td></tr><tr align=middle><td>"
                .format(keyframeFilenameInHTML), file=fout
            )

        print(
            "<font size=1>{}:{}</font></td></tr></table></td>\n"
            .format(i, tmp[2]), file=fout
        )
        #  set_trace()
        
        if i % 10 == 9:
            print("</tr></table>\n \n", file=fout)


print("</body></html>", file=fout)

fout.close()

print(">> # of retrieved correct shots = {}".format(nbCorrectShotsRetrieved))

