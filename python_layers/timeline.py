from collections import OrderedDict

CHILDCHILD = "childchild"
CHILDROBOT = "childrobot"


GOALORIENTED="goaloriented"
AIMLESS="aimless"
ADULTSEEKING="adultseeking"
NOPLAY="noplay"

TASKENGAGEMENT=(GOALORIENTED,
                AIMLESS,
                ADULTSEEKING,
                NOPLAY)

SOLITARY="solitary"
ONLOOKER="onlooker"
PARALLEL="parallel"
ASSOCIATIVE="associative"
COOPERATIVE="cooperative"

SOCIALENGAGEMENT=(SOLITARY,
                  ONLOOKER,
                  PARALLEL,
                  ASSOCIATIVE,
                  COOPERATIVE)

PROSOCIAL="prosocial"
ADVERSARIAL="adversarial"
ASSERTIVE="assertive"
FRUSTRATED="frustrated"
PASSIVE="passive"

SOCIALATTITUDE=(PROSOCIAL,
                ADVERSARIAL,
                ASSERTIVE,
                FRUSTRATED,
                PASSIVE)

CONSTRUCTS = (TASKENGAGEMENT, SOCIALENGAGEMENT, SOCIALATTITUDE)

MISSINGDATA="missingdata"


class Timeline:

    def __init__(self, construct, annotations):

        self.construct = construct

        self.timeline=OrderedDict()
        self.start = 0
        self.end = 0

        self.prepare(annotations)

    def prepare(self, annotations):
        
        for a in annotations:
            for k,v in a.items():
                if k in self.construct:
                    self.timeline[v[0]] = (v[0], v[1], k)

                    if self.start == 0:
                        self.start = v[0]
                    self.end = v[1]



    def attime(self, t):
        for k, v in self.timeline.items():
            start, end, construct = v
            if t >= start and t < end:
                return construct
        return MISSINGDATA

    def __repr__(self):
        return "timeline from %f to %f (%d sec), %s" % (self.start, self.end, self.end-self.start, str(self.construct))

if __name__=="__main__":

    import yaml
    import sys

    with open(sys.argv[1], 'r') as yml:
        coder1 = yaml.load(yml)

    for construct in CONSTRUCTS:
        timeline = Timeline(construct, coder1["purple"])
        print(timeline)

