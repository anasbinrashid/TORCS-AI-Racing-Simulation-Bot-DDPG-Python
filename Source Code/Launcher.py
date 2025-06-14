

# for Python3-based torcs python robot client
import socket
import sys
import getopt
import os
import time
PI= 3.14159265359

data_size = 2**17

# Initialize help messages
ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,
        vision=False, process_id=None, race_config_path=None, race_speed=1.0,
        rendering=True, damage=False, lap_limiter=2, recdata=False,
        noisy=False, rec_index=0, rec_episode_limit=1, rec_timestep_limit=3600,
        rank=0):
        # If you don't like the option defaults,  change them here.
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 1000000000  # 50steps/second
        self.parse_the_command_line()
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d

        #Raceconfig compat
        self.torcs_process_id = process_id
        self.race_config_path = race_config_path
        self.race_speed = race_speed
        self.rendering = rendering
        self.damage = damage
        self.lap_limiter = lap_limiter
        self.recdata = recdata
        self.noisy = noisy
        self.rec_timestep_limit = rec_timestep_limit
        self.rec_episode_limit = rec_episode_limit
        self.rec_index = rec_index
        self.rank = rank

        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()


    def setup_connection(self):
        # == Set Up UDP Socket ==
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        # == Initialize Connection To Server ==
        self.so.settimeout(1)

        n_fail = 5
        while True:
            # This string establishes track sensor angles! You can customize them.
            #a= "-90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90"
            # xed- Going to try something a bit more aggressive...
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    #Kill eventually existing process of the current TorcsEnv process
                    if self.torcs_process_id is not None:
                        try:
                            p = psutil.Process( self.torcs_process_id)
                            #Kill children... yes
                            for pchild in p.children():
                                pchild.terminate()
                            #Then kill itself
                            p.terminate()
                        except Exception:
                            self.torcs_process_id = None
                        #Sad life to be a process

                    # if self.randomisation:
                    #     self.randomise_track()

                    args = ["torcs", "-nofuel", "-nolaptime",
                        "-a", str( self.race_speed)]

                    if self.damage:
                        args.append( "-nodamage")

                    if self.noisy:
                        args.append( "-noisy")

                    if self.vision:
                        args.append( "-vision")

                    if not self.rendering:
                        args.append( "-T") # Run in console

                    #Make sure that self.race_config_path is really set
                    if self.race_config_path is not None:
                        args.append( "-raceconfig")
                        # args.append( "\"" + race_config_path + "\"")
                        args.append( self.race_config_path)

                    if self.recdata:
                        args.append( "-rechum %d" % self.rec_index)
                        args.append( "-recepisodelim %d" % self.rec_episode_limit)
                        args.append( "-rectimesteplim %d" % self.rec_timestep_limit)

                    args.append("&")

                    # print( "##### DEBUG: Args in Snake oil reset")
                    # print( args)

                    #This PID must be recovered from the Client too later
                    self.torcs_process_id = subprocess.Popen( args, shell=False).pid

                    # print("relaunch torcs")
                    # os.system('pkill torcs')
                    # time.sleep(1.0)
                    # if self.vision is False:
                    #     os.system('torcs -nofuel -nodamage -nolaptime &')
                    # else:
                    #     os.system('torcs -nofuel -nodamage -nolaptime -vision &')
                    #
                    # time.sleep(1.0)
                    # os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                       opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.so: return
        sockdata= str()

        while True:
            try:
                # Receive server data
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
                #print "Waiting for data on %d.............." % self.port
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. "+
                        "You were in %d place.") %
                        (self.port,self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                # What do I do here?
                print("Server has restarted the race on %d." % self.port)
                # I haven't actually caught the server doing this.
                self.shutdown()
                return
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H") # Clear for steady output.
                    print(self.S)
                break # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())
        # Or use this for plain output:
        #if self.debug: print self.R

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.so.close()
        self.so = None
        # sys.exit() # No need for this really.

    def randomise_track():
        # print( "### DEBUG: randomisation requested")
        # print( "### DEBUG: Profile Reuse Count", self.profile_reuse_count)
        # print( "### DEBUG: Profile resue ep", profile_reuse_ep)

        if self.profile_reuse_count == 0 or self.profile_reuse_count % self.profile_reuse_ep == 0:
            print( "### DEBUG: Generating new profile")
            track_length = 2700 # Extract form torcs maybe
            max_pos_length = int(.7 * track_length) # Floor to 100 tile
            agent_init = random.randint(0,20) * 10
            bot_count = random.randint(1,10)
            min_bound = agent_init + 50
            max_leap = math.floor((max_pos_length - min_bound) / bot_count / 100) * 100
            bot_init_poss = []
            for _ in range(bot_count):
                bot_init_poss.append( random.randint( min_bound, min_bound + max_leap))
                # Random generate in range minbound and max pos length with max leap
                min_bound += max_leap

            # Check for random config file folder and create if not exists
            randconf_dir = os.path.join(  os.path.dirname(os.path.abspath(__file__)),
                "rand_raceconfigs")
            if not os.path.isdir(randconf_dir):
                os.mkdir(randconf_dir)
            randconf_filename = "agent_randfixed_%d" % agent_init
            for bot_idx in bot_init_poss:
                randconf_filename += "_%d" % bot_idx
            randconf_filename += ".xml"
            if not os.path.isfile( os.path.join( randconf_dir, randconf_filename)):
                # Create Fielk config based on xml template
                tree = None
                root = None
                with open( os.path.join( randconf_dir, "agent_randfixed_tmplt.xml")) as tmplt_f:
                    tree = ET.parse( tmplt_f)
                    root = tree.getroot()

                driver_node = None

                driver_section = root.find(".//section[@name='Drivers']")
                driver_section.append( ET.Element( "attnum",
                    { "name": "maximum_number", "val": "%d" % (1+ bot_count)}))
                driver_section.append( ET.Element( "attstr",
                    { "name": "focused module", "val": "scr_server1" }))
                driver_section.append( ET.Element( "attnum",
                    { "name": "focused idx", "val": "1" }))

                # # Add Scr Server
                agent_section = ET.Element( "section",
                    { "name": "%d" % (1)})
                agent_section.append( ET.Element( "attnum",
                    { "name": "idx", "val": "%d" % (0) }))
                agent_section.append( ET.Element( "attstr",
                    { "name": "module", "val": "scr_server" }))
                driver_section.append( agent_section)

                driver_section.append( ET.Element( "attnum",
                    { "name": "initdist_%d" % (1), "val": "%d" % agent_init}))

                for bot_idx, bot_init_pos in enumerate( bot_init_poss):
                    bot_section = ET.Element( "section",
                        { "name": "%d" % (2+bot_idx)})
                    bot_section.append( ET.Element( "attnum",
                        { "name": "idx", "val": "%d" % (2+bot_idx) }))
                    bot_section.append( ET.Element( "attstr",
                        { "name": "module", "val": "fixed" }))
                    driver_section.append( bot_section)
                    driver_section.append( ET.Element( "attnum",
                        { "name": "initdist_%d" % (bot_idx+1), "val": "%d" % bot_init_pos}))

                print( randconf_filename)
                randconf_abspath = os.path.join( randconf_dir, randconf_filename)
                tree.write( randconf_abspath)

                self.race_config_path = randconf_abspath
                self.profile_reuse_count = 1

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        # Comment the next line for raw output:
        return self.fancyout()
        # -------------------------------------
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out= str()
        sensors= [ # Select the ones you want in the order you want them.
        #'curLapTime',
        #'lastLapTime',
        'stucktimer',
        #'damage',
        #'focus',
        'fuel',
        #'gear',
        'distRaced',
        'distFromStart',
        #'racePos',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ',
        'speedY',
        'speedX',
        'targetSpeed',
        'rpm',
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]

        #for k in sorted(self.d): # Use this to get all sensors.
        for k in sensors:
            if type(self.d.get(k)) is list: # Handle list type data.
                if k == 'track': # Nice display for track sensors.
                    strout= str()
                 #  for tsensor in self.d['track']:
                 #      if   tsensor >180: oc= '|'
                 #      elif tsensor > 80: oc= ';'
                 #      elif tsensor > 60: oc= ','
                 #      elif tsensor > 39: oc= '.'
                 #      #elif tsensor > 13: oc= chr(int(tsensor)+65-13)
                 #      elif tsensor > 13: oc= chr(int(tsensor)+97-13)
                 #      elif tsensor >  3: oc= chr(int(tsensor)+48-3)
                 #      else: oc= '_'
                 #      strout+= oc
                 #  strout= ' -> '+strout[:9] +' ' + strout[9] + ' ' + strout[10:]+' <-'
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # Nice display for opponent sensors.
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
            else: # Not a list type of value.
                if k == 'gear': # This is redundant now since it's part of RPM.
                    gs= '_._._._._._._._._'
                    p= int(self.d['gear']) * 2 + 2  # Position
                    l= '%d'%self.d['gear'] # Label
                    if l=='-1': l= 'R'
                    if l=='0':  l= 'N'
                    strout= gs[:p]+ '(%s)'%l + gs[p+3:]
                elif k == 'damage':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,10000,50,'~'))
                elif k == 'fuel':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,100,50,'f'))
                elif k == 'speedX':
                    cx= 'X'
                    if self.d[k]<0: cx= 'R'
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-30,300,50,cx))
                elif k == 'speedY': # This gets reversed for display to make sense.
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k]*-1,-25,25,50,'Y'))
                elif k == 'speedZ':
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-13,13,50,'Z'))
                elif k == 'z':
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k],.3,.5,50,'z'))
                elif k == 'trackPos': # This gets reversed for display to make sense.
                    cx='<'
                    if self.d[k]<0: cx= '>'
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k]*-1,-1,1,50,cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout= '%3d %s' % (self.d[k], bargraph(self.d[k],0,300,50,"'"))
                    else: strout= 'Not stuck!'
                elif k == 'rpm':
                    g= self.d['gear']
                    if g < 0:
                        g= 'R'
                    else:
                        g= '%1d'% g
                    strout= bargraph(self.d[k],0,10000,50,g)
                elif k == 'angle':
                    asyms= [
                          # Line 564
                        "---  ", ".__  ", "-._  ", "'-.  ", "'\\.  ", "'|.  ",
                        # Line 566
                        "  ---", "  --.", "  -._", "  -..", "  '\\.", "  '|." ]
                    rad= self.d[k]
                    deg= int(rad*180/PI)
                    symno= int(.5+ (rad+PI) / (PI/12) )
                    symno= symno % (len(asyms)-1)
                    strout= '%5.2f %3d (%s)' % (rad,deg,asyms[symno])
                elif k == 'skid': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    skid= 0
                    if frontwheelradpersec:
                        skid= .5555555555*self.d['speedX']/frontwheelradpersec - .66124
                    strout= bargraph(skid,-.05,.4,50,'*')
                elif k == 'slip': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    slip= 0
                    if frontwheelradpersec:
                        slip= ((self.d['wheelSpinVel'][2]+self.d['wheelSpinVel'][3]) -
                              (self.d['wheelSpinVel'][0]+self.d['wheelSpinVel'][1]))
                    strout= bargraph(slip,-5,150,50,'@')
                else:
                    strout= str(self.d[k])
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
       self.actionstr= str()
       # "d" is for data dictionary.
       self.d= { 'accel':0.2,
                   'brake':0,
                  'clutch':0,
                    'gear':1,
                   'steer':0,
                   'focus':[-90,-45,0,45,90],
                    'meta':0
                    }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        self.d['clutch']= clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear']= 0
        if self.d['meta'] not in [0,1]:
            self.d['meta']= 0
        if type(self.d['focus']) is not list or min(self.d['focus'])<-180 or max(self.d['focus'])>180:
            self.d['focus']= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list:
                out+= '%.3f' % v
            else:
                out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out
        return out+'\n'

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out= str()
        od= self.d.copy()
        od.pop('gear','') # Not interesting.
        od.pop('meta','') # Not interesting.
        od.pop('focus','') # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout=''
                strout= '%6.3f %s' % (od[k], bargraph(od[k],0,1,50,k[0].upper()))
            elif k == 'steer': # Reverse the graph to make sense.
                strout= '%6.3f %s' % (od[k], bargraph(od[k]*-1,-1,1,50,'S'))
            else:
                strout= str(od[k])
            out+= "%s: %s\n" % (k,strout)
        return out

# == Misc Utility Functions
def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


def drive_example(c):
    # This is a simple example of a driver that uses the server state
    S, R = c.S.d, c.R.d
    target_speed = 300
    
    # Variables to track stuck state
    stuck_speed_threshold = 3  # Speed below which we consider car to be potentially stuck
    stuck_duration_threshold = 100  # Number of iterations to consider the car stuck
    
    # Add stuck counter as a static variable if it doesn't exist
    if not hasattr(drive_example, "stuck_counter"):
        drive_example.stuck_counter = 0
    
    # Add recovery state as a static variable if it doesn't exist
    if not hasattr(drive_example, "in_recovery"):
        drive_example.in_recovery = False
    
    # Add recovery step counter as a static variable if it doesn't exist
    if not hasattr(drive_example, "recovery_steps"):
        drive_example.recovery_steps = 0
    
    # Stuck detection and recovery logic
    if S['speedX'] < stuck_speed_threshold and abs(S['speedX']) < stuck_speed_threshold:
        drive_example.stuck_counter += 1
    else:
        drive_example.stuck_counter = 0
        
    # Check if we're stuck or already in recovery mode
    if drive_example.stuck_counter > stuck_duration_threshold or drive_example.in_recovery:
        drive_example.in_recovery = True
        drive_example.recovery_steps += 1
        
        # Recovery procedure with phases
        if drive_example.recovery_steps < 50:  # Phase 1: Stop first
            R['steer'] = 0
            R['brake'] = 1
            R['accel'] = 0
            R['gear'] = 0  # Neutral
            print("RECOVERY: Stopping car...")
            
        elif drive_example.recovery_steps < 100:  # Phase 2: Reverse
            R['steer'] = -S['angle'] * 0.5  # Counter-steer to straighten while reversing
            R['gear'] = -1  # Reverse gear
            R['accel'] = 1  # Moderate acceleration
            R['brake'] = 0
            print("RECOVERY: Reversing...")
            
        elif drive_example.recovery_steps < 120:  # Phase 3: Stop again
            R['steer'] = 0
            R['brake'] = 1
            R['accel'] = 0
            R['gear'] = 0  # Neutral
            print("RECOVERY: Stopping after reversing...")
            
        else:  # Phase 4: Return to normal driving
            R['gear'] = 1
            drive_example.in_recovery = False
            drive_example.stuck_counter = 0
            drive_example.recovery_steps = 0
            print("RECOVERY: Returning to normal driving")
        
        #return  # Skip normal driving logic while in recovery
    
    # Normal driving logic (only executed when not in recovery)
    
    # Steer To Corner
    R['steer'] = S['angle'] * 10 / PI
    # Steer To Center
    R['steer'] -= S['trackPos'] * 0.10

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer'] * 50):
        R['accel'] += 0.05
    else:
        R['accel'] -= 0.01
    if S['speedX'] < 10:
        R['accel'] += 1 / (S['speedX'] + 0.1)

    # Traction Control System
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= 0.2

    # Automatic Gear Change Logic
    if S['rpm'] > 6000:
        R['gear'] = min(7, R['gear'] + 1)
    
    if S['rpm'] < 3000:
        # Don't shift down from neutral or reverse
        if R['gear'] > 1:
            R['gear'] = R['gear'] - 1
            
    # Make sure we're not still in neutral
    if S['rpm'] > 3000 and R['gear'] <= 0:
        R['gear'] = 1
        
    return

# ================ MAIN ================
if __name__ == "__main__":
    C= Client(p=3001)
    for step in range(C.maxSteps,0,-1):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()
