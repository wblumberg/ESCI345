import numpy as np
from pylab import *
import multiprocessing
import pandas as pd

"""
    2-D Monte Carlo Radiative Transfer Model
    Author: Greg Blumberg
    Email: wblumberg@ou.edu

    This code is a simple Monte Carlo Radiative Transfer Model for 
    1-layer of the atmosphere.  Photons enter the layer from the top
    at a specified angle.  Once inside the layer, photons are either scattered
    semi-randomly (an asymmetry scattering parameter is included), absorbed,
    or the photon passes completely through the layer.  In some cases,
    the photon will exit the top of the layer.

    These events are counted and output into the simulation_output.txt file.
    If the user wishes, plots showing all of the photon paths can be output.

    The behavior of the model can be changed by modifying 4 parameters:
    1.) The optical depth - impacts how optically thin or thick the cloud is 
                            a value of 10 is pretty thick.
    2.) Single Scatter Albedo (SSA) - a parameter that specifies the ratio of scattering efficiency to total extinction
                                      if set to 0, extinction is only due to absorption.
                                      if set to 1, extinction is only due to scattering.
    3.) Angle of approach - the angle at which the photon enters the layer.
    4.) Asymmetry scattering parameter (g) - a parameter that describes which direction photons are preferentially scattered towards
                                             if set to 1, all scattering is forwards.
                                             if set to 0, scattering is isotropic (all directions are likely)
                                             if set to -1, all scattering is backwards.

    The configuration can be changed by modifying the variables in namelist.py

    Future changes to the code may include:
    1.) Providing the capability to create a multi-layer atmosphere.
    2.) Providing the opportunity to place an instrument to collect photons at the top or bottom of the layer.
    3.) A connection to some Python Mie code I found.
"""   

class Photon:
    def __init__(self, theta_0, phi_0, weight=1, tau=0, scat_events=0, scattered=False):
        # tau - location in layer
        # theta_0 - angle of initial photon direction
        # phi_0 - other angle of inital photon direction

        self.tau = tau
        self.k = self.calculate_k_from_angle(np.radians(theta_0), np.radians(phi_0))
        #print self.k, theta_0
        #stop
        self.k_0 = self.k # Initial propagation vector in model coordinates
        self.weight = weight
        self.scattered = scattered
        self.scattering_events = scat_events # number of scattering events
        self.phi = phi_0 # in degrees
        self.theta = theta_0 # in degrees
        self.position = []
        self.position.append([0,0])
        self.x = 0

    def calculate_k_from_angle(self, theta, phi):
        # Returns the scattered vector omega prime in photon coordinates
        # from http://sleet.aos.wisc.edu/~gpetty/gpetty/aos640/photon_geometry.pdf

        k_p = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        return k_p

    def henyey_greenstein_inv(self, g, r):
        # returns the scattering angles using a g
        #r = np.random.uniform(0,1,1)
        if g != 0:
            num = ( 1. - np.power(g, 2) )
            den = ( 1. + ( g * ( (2. * r) - 1.) ) )
            costheta = (1. / (2.*g)) * (1. + np.power(g, 2) - np.power( num / den, 2 ))
        else:
            costheta = np.random.uniform(-1,1,1)[0]
        theta = np.arccos(costheta)
        phi = 2. * np.pi * np.random.uniform(0,1,1)[0]
        return theta, phi

    def calculate_x_p(self):
        denom = np.hypot(self.k[1], self.k[0])
        if np.abs(denom) < 0.00001:
            x_p = np.array([-np.sin(np.radians(self.phi)), np.cos(np.radians(self.phi)), 0])
        else:
            x_p = (1./denom) * np.array([-1 * self.k[1], self.k[0], 0])
        return x_p

    def scatter(self, g):
        # Sets the photon's attirbute of "scattered" to True
        # This function is called when the photon is scattered
        # This function also uses random numbers to determine the new direction
        # for the photon.
        # It will change self.k to a new value.
        #print self.k
        #print "Scattering!"
        self.scattering_events += 1
        self.scattered = True
        r = np.random.uniform(0,1,1)[0]
        theta, phi = self.henyey_greenstein_inv(g, r)
        #print theta, g
        x_p = self.calculate_x_p()
        k_p = self.calculate_k_from_angle(theta, phi)
        new_k = np.array([(x_p[0] * k_p[0]) - (self.k[2] * x_p[1] * k_p[1]) + (self.k[0] * k_p[2]),\
                          (x_p[1] * k_p[0]) + (self.k[2] * x_p[0] * k_p[1]) + (self.k[1] * k_p[2]),\
                          (k_p[1] * self.k[0] * x_p[1]) - (k_p[1] * self.k[1] * x_p[0]) + (self.k[2] * k_p[2])])
        self.k = new_k
        #print self.k
        self.theta = np.degrees(np.arccos(-self.k[2]))
        #print self.theta, self.k
        self.phi = np.degrees(np.arctan2(self.k[1], self.k[0]))
        #self.phi = np.degrees(np.arcsin(self.k[1]/np.sin(np.radians(self.theta))))[0]
        #if ~np.isfinite(self.phi):
        #    self.phi = 0.0
        #print self.theta, self.phi
        #print self.k

    def exittop(self):
        # Photon has exited the top of the layer
        return -2

    def exitbase(self):
        # Photon has exited the base of the layer
        return -1

    def absorbed(self):
        # Logic for the absorbed extiniction event
        return 0

    def movePhoton(self, layer_obj):
        # moves the photon within the layer
        # It uses a random number to determine the optical distance
        # the photon will travel before its next extinction event.
        # It updates self.tau to the new vertical position in the layer
        #print "Move the Photon."
        r = np.random.uniform(0,1,2) # movement, scatter?
        
        #print r
        # Calculate new tau of the Photon
        random = -1. * np.log(1 - r[0])
        tau_prime = random * self.k[2]
        #print np.cos(np.radians(self.theta))
        #print "Beer's law, tau_prime, self.tau"
        #print np.exp(-layer_obj.getLayerOD()/self.k[2]), tau_prime, self.tau
        x_prime = random * self.k[0]
        
        self.tau += tau_prime
        self.x += x_prime
            
        if type(self.x) == np.ndarray:
            self.position.append([self.x[0], self.tau[0]])
            self.x = self.x[0]
            self.tau = self.tau[0]
        else:
            self.position.append([self.x, self.tau])

        #print "PHOTON TAU:", self.tau
        # Find out whether or not Photon has left the top or bottom of the layer
        if self.tau >= layer_obj.getLayerOD():
            # Photon has exited bottom of layer
            #print "Photon has exited bottom of the layer."
            
            # TODO: create a random probablity that the photon will be reflected assuming a Lambertian surface
            #       this means that the new direction of the photon is completely random and that there is a
            #       uniform probability of the different directions the photon could go.
            #       The new k vector does not depend upon the current k vector at all.
            #
            #   http://stackoverflow.com/questions/14476973/calculating-diffuse-lambertian-reflection
            #print "PHOTON IS GONNA EXIT."
            #stop
            r = np.random.uniform(0,1,1)[0] # Reflect upwards or absorb?
            if r < layer_obj.sfc_albedo:
                self.phi = 2. * np.pi * r # new scattering angle phi
                self.theta = np.random.uniform(-90,90,1)
                #print np.cos(np.radians(self.theta))
                self.k = self.calculate_k_from_angle(np.radians(self.theta), self.phi)
                self.k[2] = self.k[2] * -1
                # position photon at the boundary
                self.x = 0
                self.tau = layer_obj.getLayerOD() - 0.00001
                self.position = self.position[:len(self.position)-1]
                self.position.append([self.x, self.tau])   
                event_flag = 1
                return event_flag
            else:
                #print "PHOTON GOT ABSORBED BY SURFACE."
                return self.exitbase()
        
        elif self.tau <= 0:
            #print "Photon has exited top of the layer."
            # Photon has exited top of layer
            return self.exittop()

        # If not, it's encountered an extinction event...is it scattering or absorption
        if r[1] <= layer_obj.getSSA():
            #print "Photon has been scattered."
            # Photon has been scattered
            self.scatter(layer_obj.getG())
            event_flag = 1
            #print "pos:", self.x, self.tau
            #print type(self.x), type(self.tau[0])
            #print self.position
        else:
            #print "Photon was absorbed."
            return self.absorbed()
        
        return event_flag

    def position_vector(self):
        return np.asarray(self.position)

class Layer:
    def __init__(self, layer_tau, g, ssa, sfc_albedo=0):
        # layer_tau is the layer optical depth to figure out whether or not
        # photon has exited layer.
        # g = the asymmetry parameter for the layer (unitless)
        # ssa = the single scatter albedo (unitless)
        if g > 1 or g < -1:
            raise ValueError(f"Invalid value of asymmetry parameter: {g}.  Value must be between -1 and 1.")
        if layer_tau < 0:
            raise ValueError(f"Invalid value of layer optical depth: {layer_tau}.  Value must be greater than 0.")
        if ssa < 0 or ssa > 1:
            raise ValueError(f"Invalid value of single scatter albedo: {ssa}.  Value must be between 0 and 1.")
        if sfc_albedo:
            raise ValueError(f"Invalid value of surface albedo: {sfc_albedo}.  Value must be between 0 and 1.")
        self.g = g
        self.layer_tau = layer_tau
        self.ssa = ssa
        self.sfc_albedo = sfc_albedo

    def getSSA(self):
        return self.ssa

    def getLayerOD(self):
        return self.layer_tau

    def getG(self):
        return self.g

class Instrument:
    def __init__(self, place):
        # Place can either be top or bottom
        self.place = place
        self.fov = 0.4 # an angle specifying the field of view (a subportion of an angle)

class Atmosphere:
    def __init__(self, layers):
        self.layers = layers # a list of Layer objects.
        self.layer_photon_is_in = 0

def run_photon(i):
    scattering_events = 0
    absorption_events = 0
    pure_absorption_events = 0
    toa_events = 0
    boa_events = 0
    direct_beam = 0
    diffuse_beam = 0
 
    #print "Initializing new photon..."
    photon = Photon(i[2],0)
    #print "\nNew Photon:"
    #print "Photon: ", i[0]
    #print "\n\n\n"
    while True:
        #print "Moving photon..."
        move = photon.movePhoton(i[1])
                
        if move < 1:
            #print "Photon died...\n"
            break
        if photon.scattered is True:
            scattering_events += 1
    
    vec = photon.position_vector()
    #print(scattering_events)
    plot(vec[:,0], vec[:,1], 'k-', lw=0.5)
    if move == -2:
        toa_events += 1
        plot([vec[-1,0]], [vec[-1,1]], 'mo', ms=scattering_events*2)
    elif move == -1:
        boa_events += 1
    elif move == 0:
        absorption_events += 1
        plot([vec[-1,0]], [vec[-1,1]], 'bo', ms=scattering_events*2)
    if move == 0 and photon.scattered is False:
        pure_absorption_events += 1 # Photon gets absorbed in the layer
    if move == -1 and photon.scattered is False:
        direct_beam += 1 # Photon reached the bottom of the atmosphere w/o any scattering
        plot([vec[-1,0]], [vec[-1,1]], 'go', fillstyle='none')
    if move == -1 and photon.scattered is True:
        diffuse_beam += 1 # Photon reached the bottom of the atmosphere, but scattering occured
        plot([vec[-1,0]], [vec[-1,1]], 'ro', ms=scattering_events*2)
        
    return [direct_beam, diffuse_beam, toa_events, absorption_events, scattering_events]

def runSimulation(tau, g, ssa, theta_0, total_photons=10, sfc_albedo=0, plot_paths=True):
    lyr = Layer(tau, g, ssa,sfc_albedo) # Case 3
    scattering_events = 0
    absorption_events = 0
    pure_absorption_events = 0
    toa_events = 0
    boa_events = 0
    direct_beam = 0
    diffuse_beam = 0
    #pool = multiprocessing.Pool(processes=num_processors)
    args = []
    for i in np.arange(0, total_photons,1):
        arg = [i, lyr, theta_0]
        args.append(run_photon(arg))
    #print len(args)
    #results = pool.map(run_photon, args)
    #results = np.sum(results, axis=0)
    args = np.asarray(args)

    scattering_counts = np.histogram(args[:,-1], bins=[0,1,2,3,1000000])[0]
    results = np.sum(args, axis=0)
    ylim(-1, lyr.getLayerOD() + 1)
    gca().invert_yaxis()
    axhline(y=0, color='k', linewidth=3)
    axhline(y=lyr.getLayerOD(), color='k', linewidth=3)
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

    msizes = np.array([1, 2, 3]) * 2

    l1, = plt.plot([],[], 'or', markersize=msizes[0])
    l2, = plt.plot([],[], 'or', markersize=msizes[1])
    l3, = plt.plot([],[], 'or', markersize=msizes[2])
    #l4, = plt.plot([],[], 'or', markersize=msizes[3])
    
    labels = [f'1 ({scattering_counts[1]:d})', f'2 ({scattering_counts[2]:d})', f'3 ({scattering_counts[3]:d})']
    
    leg = plt.legend([l1, l2, l3], labels, ncol=1, frameon=True, fontsize=12,
    handlelength=2, loc = 1, borderpad = .2,
    handletextpad=1, title='# Scat. Events', scatterpoints = 1)
    
    if plot_paths is True:
        title(r"Simulation Parameters: $\tau=$" + str(tau) + ", $g=$" + str(g) + ", $\omega_0=$" + str(ssa) + ", $\Theta_0=$" + str(theta_0))
        #savefig('OD' + str(tau) + '_g' + str(g) + '.ssa' + str(ssa) + '_ang' + str(theta_0) + '.png', bbox_inches='tight')
        show()
    else:
        clf()
    return results[0], results[1], results[2], results[3], scattering_counts[0], scattering_counts[1], scattering_counts[2], scattering_counts[3]

def mc_rtm(taus=[1,0.5,0], g=[-1,0,1], ssas=[1,0.99,0.5], theta_0s=[10,90,70], num_photons=10):
    #if len(taus) == len(theta_0s) == len(ssas) == len(g):
    #    print("The list of experiments in namelist.py are not the same length.")
    #    print("Can't run experiments, quitting.")
    #    return
    
    cases = np.arange(1,len(taus)+1,1)

    num_photons=float(num_photons)

    all_cases = []
    for i in cases:
        print("SIMULATION #:", i)
        print("------------------------------------------------------------------")
        d = np.array(runSimulation(taus[i-1], g[i-1], ssas[i-1], theta_0s[i-1], sfc_albedo=0, total_photons=num_photons))/num_photons
        results = d[:4]#/num_photons
        scattering_counts = d[4:]
        #results = np.concatenate((results, scattering_counts))
        print("------------------------------------------------------------------")
        print("Zenith Angle (\u03b8):", theta_0s[i-1])
        print("Optical Depth (\u03c4):", taus[i-1])
        print("Single Scatter Albedo (\u03c9):", ssas[i-1])
        print("Asymmetry Parameter (g):", g[i-1])
        print("Beer's Law Transmission (t):", np.exp( - taus[i-1]/np.cos(np.radians(theta_0s[i-1])) ))
        print("------------------------------------------------------------------")
        print("SIMULATION RESULTS:")
        titles = ['Direct Beam', 'Diffuse Beam', 'TOA Events', 'Absorption']
        row_format = "{:>2}" + "{:>15}" * (len(titles))
        print(row_format.format("", *titles))
        print(row_format.format("", *results))
        print()
        titles = ['No Scat', '1 Scat', '2 Scat', '>3 Scat']
        row_format = "{:>3}" + "{:>10}" * (len(titles))
        print("# Photons w/" + row_format.format("", *titles))
        print("            " + row_format.format("", *scattering_counts))
        print("------------------------------------------------------------------")
        print()
        
        all_cases.append([theta_0s[i-1], taus[i-1], ssas[i-1], g[i-1], np.exp( - taus[i-1]/np.cos(np.radians(theta_0s[i-1]))),*results, *scattering_counts])
        
    df_cases = pd.DataFrame(all_cases, columns = ['Zenith Angle', 'Optical Depth', 'SSA', 'g', 'trans', 'Direct Beam', 'Diffuse Beam', 'TOA Events', 'Absorption', 'No Scat', '1 Scat', '2 Scat', '>3 Scat'])
    return df_cases
