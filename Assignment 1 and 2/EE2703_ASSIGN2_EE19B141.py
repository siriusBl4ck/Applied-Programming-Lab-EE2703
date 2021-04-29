# MySPICE v0.1 by Saurav Sachin Kale EE19B141
# EE2703 : Applied Programming Lab

# Usage instructions:

# Command to run:
# python3 EE2703_ASSIGN2_EE19B141.py <path_to_netlist_file>

# If your file is myCkt.netlist, include the ".netlist" part in the file path, otherwise MySPICE will refuse to parse it.
# MySPICE only parses files with the extension .netlist

# If the py file and the netlist file are in the same directory, example of correct command is:
# python3 EE2703_ASSIGN2_EE19B141.py myCkt.netlist

# supports solving one circuit per netlist file. Do not define multiple circuits in same file
# sources are in form  NAME n1 n2 VALUE, n1 -> positive terminal, n2 -> negative terminal

from sys import argv, exit
import numpy as np
import math

print("MySPICE v0.1")

# definitions of keywords
CKT_BEGIN = ".circuit"
CKT_END = ".end"
CKT_FREQ = ".ac"
COM_RESISTOR = "R"
COM_INDUCTOR = "L"
COM_CAPACITOR = "C"
SRC_V = "V"
SRC_C = "I"
SRC_DC = "dc"
SRC_AC = "ac"
DEP_SRC_VCVS = "E"
DEP_SRC_VCCS = "G"
DEP_SRC_CCVS = "H"
DEP_SRC_CCCS = "F"

DC_APPROX = 1e-50

# each basic component will have the following attributes
class Component:
    type = None
    name = None
    src_isAC = False
    # ports[0] is considered to be the negative terminal, ports[1] is considered to be positive terminal
    ports = []
    dependencies = []
    value = None
    def printInfo(self, reverse = False):
        if reverse:
            print(self.value, reversed(self.dependencies), reversed(self.ports), self.name)
        else:
            print(self.name, self.ports, self.dependencies, self.value)
    def __init__(self, _name = None, _ports = [], _dependencies = [], _value = None, _isAC = False):
        self.type = _name[0]
        self.name = _name
        self.ports = _ports
        self.dependencies = _dependencies
        self.value = _value
        self.src_isAC = _isAC

#circuit will be a list of components
class Circuit:
    ckt = []
    freq = 0
    isAC = False
    def printCkt(self):
        for i in range(len(self.ckt)):
            self.ckt[i].printInfo()
    def append(self, com):
        self.ckt.append(com)
    def size(self):
        return len(self.ckt)
    def get(self, i):
        return self.ckt[i]
    def __init__(self):
        self.freq = 0
        self.ckt = []
    def getIterator(self):
        return self.ckt

# each node concists of its name, the nodes it is adjacent to, and the components which bridge the adjacent nodes and the node itself
class Node:
    name = None
    nieghbors = []
    connectedComponents = []
    def __init__(self, _name = None):
        name = _name
        self.nieghbors = []
        self.connectedComponents = []

#function to convert complex number to polar form
def polar(z):
    a= z.real
    b= z.imag
    r = math.hypot(a,b)
    theta = np.degrees(math.atan2(b,a))
    return (r,theta) # use return instead of print.

class Solver:
    # dict containing name of node -> node object
    nodes = {}
    # dict containing name of node -> index of node in matrix
    nodesToIndices = {}
    # dict containing name of node -> index of name
    indicesToNodes = []
    # list containing all the voltage sources in the circuit
    voltageSrc = []
    # names of voltage source to index in voltageSrc
    voltagesToIndex = {}
    # the MNA matrix
    M = None
    # the RHS vector
    b = None
    # the values of voltages which we will calculate using 
    x = None
    # is the ckt ac or dc (if AC then store freq)
    freq = 0

    # takes the circuit object and extracts nodes from it
    def populateNodes(self, ckt):
        index = -1
        self.freq = ckt.freq
        # for each component in circuit, check if its port (terminal) nodes have been seen before
        # if yes, simply add the component to the node's connectedComponents attribute
        # if no, create a new node object with necessary attributes and include it in the nodes dictionary
        for com in ckt.getIterator():
            # voltage sources require special treatment, since we are to deal with the current through the voltage sources separately
            # we simply add these to a list of voltage sources (voltageSrc)
            # we also maintain a dictionary (voltagesToIndex) to store the name of the voltage source : its index in voltageSrc
            if com.type == SRC_V:
                self.voltageSrc.append(com)
                self.voltagesToIndex[com.name] = len(self.voltageSrc) - 1
            if self.nodes.get(com.ports[0]) != None:
                # print("YES!")
                self.nodes.get(com.ports[0]).nieghbors.append(com.ports[1])
                self.nodes.get(com.ports[0]).connectedComponents.append(com)
            else:
                # print("NO!")
                index += 1
                self.nodes[com.ports[0]] = Node()
                self.nodesToIndices[com.ports[0]] = index
                self.indicesToNodes.append(com.ports[0])
                self.nodes.get(com.ports[0]).name = com.ports[0]
                self.nodes.get(com.ports[0]).nieghbors.append(com.ports[1])
                self.nodes.get(com.ports[0]).connectedComponents.append(com)
            
            if self.nodes.get(com.ports[1]) != None:
                # print("YES!")
                self.nodes.get(com.ports[1]).nieghbors.append(com.ports[0])
                self.nodes.get(com.ports[1]).connectedComponents.append(com)
            else:
                # print("NO!")
                index += 1
                self.nodes[com.ports[1]] = Node()
                self.nodesToIndices[com.ports[1]] = index
                self.indicesToNodes.append(com.ports[1])
                self.nodes.get(com.ports[1]).name = com.ports[1]
                self.nodes.get(com.ports[1]).nieghbors.append(com.ports[0])
                self.nodes.get(com.ports[1]).connectedComponents.append(com)
                
        # print(self.nodesToIndices)
        
        # if no GND is defined, solving this circuit is impossible, so raise an error if it is not found in nodes dictionary
        try:
            self.nodes.get('GND')
        except:
            log("SEMANTIC", "No GND defined!!!!!")
            exit()

    def printNodes(self):
        # for debugging purposes we print the nodes
        for n in self.nodes.values():
            #print("Node", n.name)
            #print(n.nieghbors)
            if (len(n.nieghbors) <= 1):
                log("SEMANTIC", "Node " + n.name + " is connected only to one or less other node(s) (this is a useless, dangling node!)", warn=True)
    
    def populateMatrices(self):
        # we have number of variables (= no. of equations) = number of nodes + number of currents through the voltage sources
        dimensions = len(self.nodes.keys()) + len(self.voltageSrc)
        # we designate the equations in index = 0 to number of nodes - 1 as the KCL equations
        # therefore, the equations from index = number of nodes to end are the voltage src equations
        # voltageEqnIndex keeps track of the index of the next coming voltage equation
        # therefore it is initialized now to = number of nodes
        voltageEqnIndex = len(self.nodes.keys())
        # initialize M and b with the required dimensions we just calculated
        self.M = np.zeros((dimensions, dimensions), dtype='complex')
        self.b = np.zeros(dimensions, dtype='complex')

        # for each node in nodes dict
        for n in self.nodes.values():
            #print("Filling node", n.name)
            # find the index of the node. This gives us the row index of M to populate the equation with respect to node n
            index = self.nodesToIndices[n.name]
            # if the node is GND, simply V_GND = 0 is the equation, then continue. No need of anything else
            if n.name == 'GND':
                self.M[index][index] = 1
                continue
            # but if its NOT GND, then we need to take the connected components into account
            # note that since every device has two terminals, the length of the n.nieghbors, and n.connectedComponents has to be same
            # for each one of node n's neighboring nodes
            for i in range(len(n.nieghbors)):
                # find the nieghbor's index
                nieghborIndex = self.nodesToIndices[n.nieghbors[i]]
                # this is a debug message, this means that two nodes have been connectedd by a short
                # in this case, the netlist file is slightly wrong, one of the nodes is same as another
                # TODO: What happens when there is a redundant node, give warning/error, convert this eqn to Vindex - Vnieghbor = 0
                if (n.connectedComponents[i].value == 0):
                    print("hmmm divide by zero avoided in component.value :/")
                    continue
                
                # for a resistive component connecting the nodes, add the appropriate entries in their stamps in the matrix M
                if n.connectedComponents[i].type == COM_RESISTOR and n.connectedComponents[i].value != 0:
                    self.M[index][nieghborIndex] -= 1/n.connectedComponents[i].value
                    self.M[index][index] += 1/n.connectedComponents[i].value
                
                # for a capacitive component, we simply ignore if it is DC, else fill required impedance
                if n.connectedComponents[i].type == COM_CAPACITOR and n.connectedComponents[i].value != 0:
                    if self.freq != 0:
                        # impedance of capacitor = -j/WC
                        impedance = complex(0, -1/(self.freq * n.connectedComponents[i].value))
                        self.M[index][nieghborIndex] -= 1/impedance
                        self.M[index][index] += 1/impedance
                    else:
                        impedance = complex(0, -1/(DC_APPROX * n.connectedComponents[i].value))
                        self.M[index][nieghborIndex] -= 1/impedance
                        self.M[index][index] += 1/impedance
                
                # for an inductive component
                if n.connectedComponents[i].type == COM_INDUCTOR and n.connectedComponents[i].value != 0:
                    if self.freq != 0:
                        # impedance of capacitor = jWL
                        impedance = complex(0, self.freq * n.connectedComponents[i].value)
                        self.M[index][nieghborIndex] -= 1/impedance
                        self.M[index][index] += 1/impedance
                    else:
                        # steady state eqn across inductor in DC is V_index - V_neighbor = 0 (short ckt)
                        impedance = complex(0, DC_APPROX * n.connectedComponents[i].value)
                        self.M[index][nieghborIndex] -= 1/impedance
                        self.M[index][index] += 1/impedance

                # for a voltage source, add the current through that voltage source term to the KCL equation
                # here we have assumed passive sign convention for the current, that every voltage source is generating energy
                if n.connectedComponents[i].type == SRC_V:
                    if n.connectedComponents[i].ports[0] == n.name:
                        self.M[index][voltageEqnIndex + self.voltagesToIndex[n.connectedComponents[i].name]] = 1
                    else:
                        self.M[index][voltageEqnIndex + self.voltagesToIndex[n.connectedComponents[i].name]] = -1
                
                # for a current source, -i should go in the appropriate KCL equation on RHS
                # here we assume current is flowing from port[0] to port[1] of current source
                if n.connectedComponents[i].type == SRC_C:
                    print("CURRENT SRC")
                    print(n.connectedComponents[i].ports)
                    if n.connectedComponents[i].ports[0] == n.name:
                        self.b[index] = -n.connectedComponents[i].value
                    else:
                        self.b[index] = n.connectedComponents[i].value
        
        # additional voltage source relations V1 - V2 = Vsrc
        # for every voltage and current source, port[0] considered negative terminal, port[1] considered positive terminal
        for v in range(len(self.voltageSrc)):
            negativeIndex = self.nodesToIndices[self.voltageSrc[v].ports[0]]
            positiveIndex = self.nodesToIndices[self.voltageSrc[v].ports[1]]
            self.M[voltageEqnIndex + v][negativeIndex] = -1
            self.M[voltageEqnIndex + v][positiveIndex] = 1
            self.b[voltageEqnIndex + v] = self.voltageSrc[v].value
        # debugging purposes
        print(self.M)
        print(self.b)

    def solveCkt(self):
        #solve Mx = b
        try:
            self.x = np.linalg.solve(self.M, self.b)
        except:
            log("SINGULAR_MAT_", "The circuit described does not have a unique solution!!!!")
            exit()
        print()
        print("--------------")
        print()
        print("Solution is in polar form (magnitude, phase in degrees)")
        for i in range(len(self.nodes.keys())):
            if self.indicesToNodes[i] == 'GND':
                print("Voltage at node", self.indicesToNodes[i], complex(0, 0))
            else:
                print("Voltage at node", self.indicesToNodes[i], polar(-self.x[i]))
        for i in range(len(self.nodes.keys()), len(self.nodes.keys()) + len(self.voltageSrc)):
            print("Current in", self.voltageSrc[i - len(self.nodes.keys())].name, polar(self.x[i]))


#list of circuits in the file (ability to define multiple circuits in the same file, nested circuits are not allowed)
circuits = []

# extract the filepath of netlist file from command line arguments
try:
    netlistFile = argv[1]
except:
    # index 0 is "Spice1.py", index 1 should be filename, but if that isnt given, it will trigger Index out of range error
    print("ERROR : Netlist filepath not specified!")
    exit()

print("Loading file : ", netlistFile)

if (len(netlistFile) >= 8):
    if (netlistFile[len(netlistFile) - 8:] != ".netlist"):
        print("Invalid file type provided! Please enter only a .netlist file!\nMySPICE exiting...")
        exit()
else:
    print("Invalid file type provided! Please enter only a .netlist file!\nMySPICE exiting...")
    exit()

# open the file specified by the path
try:
    f = open(netlistFile)
except:
    print("ERROR: The netlist filepath you supplied is either incorrect, or no such file exists!")
    exit()

# stores all the netlist as a list of strings, each string being each line of the netlist
lines = []
try:
    lines = f.readlines()
    f.close()
except:
    print("ERROR: There was an issue in reading the file data\nChecking the file path or the file itself may help\nExiting...")
    exit()

if (len(lines) == 0):
    print("Empty File found... exiting")
    exit()

# Logging functionality
def log(code, message, line_num = None, warn=False):
    if (warn):
        print(code + "_WARNING: ", message)
    else:
        print(code + "_ERROR: ", message)
    if (line_num != None):
        print("Location line: " + str(line_num + 1) + "] " + lines[line_num])

# discards everything after the first '#' on each line (removing comments)
for i in range(len(lines)):
    if lines[i].find("#") != -1:
        lines[i] = lines[i][:lines[i].find("#")] + '\n'

"""
# Printing contents of read file
print("=======FILE CONTENTS======")
for e in lines:
    print(e, end='')
print("========END OF FILE=======")

print ("Netlist loaded successfully. Parsing...")
"""

# check if component names have been taken before, if not, add it to the dictionary
nameHash = {}
def checkNameAvailableAndAdd(name, line_num):
    if (nameHash.get(name) != None):
        log("DUPLICATE_DEFN", "Duplicate definitions for the same name detected in same circuit!")
        log("DUPLICATE_DEFN", "A component of this name has already been defined at:", nameHash[name])
        log("DUPLICATE_DEFN", "and here:", line_num)
        exit(0)
    else:
        nameHash[name] = line_num
def clearNameCache():
    nameHash.clear()

# this function checks if the values in words (list of parsed tokens) at the given indices toBeChecked are alphanumeric or not
def checkAlpha(words, toBeChecked, line_num):
    for i in toBeChecked:
        if not words[i].isalnum():
            log("INVALID_VALUE", "arguments specified are not of alphanumeric type!", line_num)
            exit()

# parse each line between CKT_BEGIN and CKT_END:
cktGiven = False
foundInvalidKeyword = False
foundBeginning = False
foundEnd = True
cktBeginLineNum = -1
numOfacDirs = 0
numOfacSrc = 0

for i in range(len(lines)):
    words = lines[i].split()

    if (len(words) == 0):
        continue

    if words[0] == CKT_BEGIN:
        foundBeginning = True
        cktGiven = True
        cktBeginLineNum = i
        # make sure there is no beginning before the end (prevents nested circuits)
        if not foundEnd:
            log("SYNTAX", "attempt to define a .circuit within a .circuit block!", i)
            exit()
        foundEnd = False
        # new circuit can have same component names as previous circuit, so clear the name cache
        clearNameCache()

        #errors related to AC of previous circuit (if this is the first ckt, numOfacDirs = numOfacSrc = 0)
        if numOfacDirs < numOfacSrc:
            log("INCOMPLETE_INFO", "the frequency of all AC sources has not been defined! (NOTE: The erroneous .ac directives are ABOVE the shown line!)", i)
            exit()
        elif numOfacDirs > numOfacSrc:
            log("PARSING", "more .ac directives encountered than voltage sources! (NOTE: The erroneous .ac directives are ABOVE the shown line!)", i)
            exit()
        
        # clear for current circuit
        numOfacDirs = 0
        numOfacSrc = 0

        #create a new circuit object, append to circuits
        circuits.append(Circuit())
        continue
    elif words[0] == CKT_END:
        foundEnd = True
        # make sure there is no end without a beginning (detects .ends without .circuits)
        if not foundBeginning:
            log("PARSING", "found a .end without a corresponding .circuit!", i)
            exit()
        foundBeginning = False
        continue
    elif words[0] == CKT_FREQ:
        if foundBeginning or (not foundEnd):
            log("PARSING", ".ac directive found within circuit definition! Define .ac after .end", i)
            # this can occur due to the user forgetting to put .end
            # this method accounts for both cases (just a regular error, and both error and missing .end)
            foundInvalidKeyword = True
            continue
        else:
            # verify .ac V_NAME FREQ
            if len(words) != 3:
                if len(words) > 3:
                    log("PARSING", "too many arguments passed for .ac directive!", i)
                    exit()
                else:
                    log("PARSING", "too few arguments passed for .ac directive!", i)
                    exit()
            if nameHash.get(words[1]) == None:
                log("SYNTAX", words[1] + " not found in circuit!", i)
                exit()
            else:
                try:
                    circuits[len(circuits) - 1].freq = 2*np.pi*float(words[2])
                    circuits[len(circuits) - 1].isAC = True
                    numOfacDirs += 1
                except:
                    log("PARSING", "specified .ac frequency not convertible to float!", i)
                    exit()

    # this is the valid part of circuit
    if foundBeginning and not foundEnd and not foundInvalidKeyword:
        # if there are no commands in the line, skip it, it probably contains only comments
        if (len(words) == 0):
            continue

        #parse the valid keywords
        if words[0][0] in [SRC_V, SRC_C]:
            # NAME n1 n2 ac/dc VALUE PHASE
            # NAME n1 n2 a/dc VALUE (assumed 0 phase)
            # NAME n1 n2 VALUE (assumed dc)
            # verify this syntax and feed it into the component object
            if len(words) == 6:
                # NAME n1 n2 ac/dc VALUE PHASE
                checkAlpha(words, [1, 2], i)
                checkNameAvailableAndAdd(words[0], i)
                if words[3] not in [SRC_AC, SRC_DC]:
                    log("PARSING", "Invalid keyword detected for ac/dc specification!", i)
                    exit()
                print(words)
                
                try:
                    name = words[0]
                    type = words[0][0]
                    ports = []
                    ports.append(words[1])
                    ports.append(words[2])
                    dependencies = []
                    amplitude = None
                    if (words[3] == SRC_AC):
                        amplitude = 0.5*float(words[4])
                    else:
                        amplitude = float(words[4])
                    phase = float(words[5])
                    value = amplitude * complex(np.cos(phase), np.sin(phase))
                    # add to ckt
                    circuits[len(circuits) - 1].append(Component(name, ports, dependencies, value))
                    if words[3] == SRC_AC:
                        circuits[len(circuits) - 1].isAC = True
                        numOfacSrc += 1
                except:
                    log("INVALID_VALUE", "the arguments specified to component are not valid!", i)
                    exit()
            elif len(words) == 5:
                # NAME n1 n2 ac/dc VALUE (assumed 0 phase)
                checkAlpha(words, [1, 2], i)
                checkNameAvailableAndAdd(words[0], i)
                if words[3] not in [SRC_AC, SRC_DC]:
                    log("PARSING", "Invalid keyword detected for ac/dc specification!", i)
                    exit()
                print(words)
                
                try:
                    name = words[0]
                    type = words[0][0]
                    ports = []
                    ports.append(words[1])
                    ports.append(words[2])
                    dependencies = []
                    amplitude = None
                    if (words[3] == SRC_AC):
                        amplitude = 0.5*float(words[4])
                    else:
                        amplitude = float(words[4])
                    value = complex(amplitude)
                    # add to ckt
                    circuits[len(circuits) - 1].append(Component(name, ports, dependencies, value))
                    if words[3] == SRC_AC:
                        circuits[len(circuits) - 1].isAC = True
                        numOfacSrc += 1
                        print("detected AC src")
                except:
                    log("INVALID_VALUE", "the arguments specified to component are not valid!", i)
                    exit()
            elif len(words) == 4:
                # NAME n1 n2 VALUE (assumed dc)
                checkAlpha(words, [1, 2], i)
                checkNameAvailableAndAdd(words[0], i)
                print(words)
                
                try:
                    name = words[0]
                    type = words[0][0]
                    ports = []
                    ports.append(words[1])
                    ports.append(words[2])
                    dependencies = []
                    value = complex(words[3])
                    # add to ckt
                    circuits[len(circuits) - 1].append(Component(name, ports, dependencies, value))
                except:
                    log("INVALID_VALUE", "the arguments specified to component are not valid!", i)
                    exit()
            else:
                if (len(words) > 6):
                    log("PARSING", "too many arguments for source!", i)
                else:
                    log("PARSING", "too few arguments for source!", i)
                exit()
        elif words[0][0] in [COM_RESISTOR, COM_CAPACITOR, COM_INDUCTOR]:
            # NAME n1 n2 VALUE
            # verify this syntax and feed it into the component object
            if len(words) != 4:
                if (len(words) > 4):
                    log("PARSING", "too many arguments for component!", i)
                else:
                    log("PARSING", "too few arguments for component!", i)
                exit()
            else:
                checkAlpha(words, [1, 2], i)
                checkNameAvailableAndAdd(words[0], i)
                # print(words)
                try:
                    name = words[0]
                    type = words[0][0]
                    ports = []
                    ports.append(words[1])
                    ports.append(words[2])
                    dependencies = []
                    value = float(words[3])
                    # add to ckt
                    circuits[len(circuits) - 1].append(Component(name, ports, dependencies, value))
                except:
                    log("INVALID_VALUE", "the arguments specified to component are not convertible to float!", i)
                    exit()
        elif words[0][0] in [DEP_SRC_VCCS, DEP_SRC_VCVS]:
            # NAME n1 n2 n3 n4 VALUE
            # verify this syntax and feed it into the component object
            if len(words) != 6:
                if (len(words) > 6):
                    log("PARSING", "too many arguments for component!", i)
                else:
                    log("PARSING", "too few arguments for component!", i)
                exit()
            else:
                checkAlpha(words, [1, 2, 3, 4], i)
                checkNameAvailableAndAdd(words[0], i)
                # print(words)
                try:
                    name = words[0]
                    type = words[0][0]
                    ports = []
                    ports.append(words[1])
                    ports.append(words[2])
                    dependencies = []
                    dependencies.append(words[3])
                    dependencies.append(words[4])
                    value = complex(words[5])
                    # add to ckt
                    circuits[len(circuits) - 1].append(Component(name, ports, dependencies, value))
                except:
                    log("INVALID_VALUE", "the arguments specified to component are not convertible to float!", i)
                    exit()
        elif words[0][0] in [DEP_SRC_CCCS, DEP_SRC_CCVS]:
            # NAME n1 n2 V... VALUE
            # verify this syntax and feed it into the component object
            if len(words) != 5:
                if (len(words) > 5):
                    log("PARSING", "too many arguments for component!", i)
                else:
                    log("PARSING", "too few arguments for component!", i)
                exit()
            else:
                checkAlpha(words, [1, 2], i)
                checkNameAvailableAndAdd(words[0], i)
                # check if voltage source is given or not
                if (words[3][0] != 'V'):
                    log("INVALID_ARGUMENT", "Invalid dependency supplied to current controlled source!", i)
                    exit()
                # print(words)
                try:
                    name = words[0]
                    type = words[0][0]
                    ports = []
                    ports.append(words[1])
                    ports.append(words[2])
                    dependencies = []
                    dependencies.append(words[3])
                    value = complex(words[4])
                    # add to ckt
                    circuits[len(circuits) - 1].append(Component(name, ports, dependencies, value))
                except:
                    log("INVALID_VALUE", "the arguments specified to component are not convertible to float!", i)
                    exit()
        else:
            log("PARSING", "No such keyword/component exists!", i)
            foundInvalidKeyword = True
            # we are not quitting the program despite the error as of now 
            # to help the user differentiate easily between the cases where they:
            # have forgotten to put the .end (means we might be processing the junk after .end)
            # or have made a genuine syntax error
            continue

# no ckt specified
if not cktGiven:
    print("WARNING: file does not contain any circuit (no .circuit found)")
# the other case is if simply have just a beginning but no end (note how this solves the problem we mentioned)
if not foundEnd:
    log("PARSING", ".end corresponding to .circuit not found!", cktBeginLineNum)
    exit()
# this statement executes only if we have a proper SPICE syntax error, in that case error message has already been printed, so exit
if foundInvalidKeyword:
    exit()

#errors related to AC of last circuit (if this is the first ckt, numOfacDirs = numOfacSrc = 0)
if numOfacDirs < numOfacSrc:
    log("INCOMPLETE_INFO", "the frequency of all AC sources has not been defined! (NOTE: The erroneous .ac directives are to be placed ABOVE the shown line!)", i)
    exit()
elif numOfacDirs > numOfacSrc:
    log("PARSING", "more .ac directives encountered than voltage sources!", i)
    exit()

# we have now populated the circuits
print("Parsing successful")
# we can traverse it in reverse order
for i in range(len(circuits)):
    print()
    print("CKT_" + str(i))
    if circuits[i].isAC:
        print("AC Angular Freq:", circuits[i].freq)
    for j in range(circuits[i].size()):
        circuits[i].get(j).printInfo()

print()
print("Solving...")

spice = Solver()

if (len(circuits) > 0):
    spice.populateNodes(circuits[0])
    spice.printNodes()
    spice.populateMatrices()
    spice.solveCkt()

print("\nMySPICE exiting")