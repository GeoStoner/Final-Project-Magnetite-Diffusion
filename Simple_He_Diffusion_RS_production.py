# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:59:58 2016

@author: ryanstoner

Model to calculate diffusion of He out of Magnetite at high diffusivities
relative to the production of He. Magnetite model from Blackburn et al. (2007)
Production term included
"""

"""
Initialize
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting up initial parameters for simple Arrhenius equation.
a = 0.0001                        # Grain size [m], 100 microns radius
D0a2 = 10**6.8                    # Diffusivity [s^-1] @ "infinite" temp. norm. 
                                  # to grain size.
Ea = 220*10**3                    # Arrhenius activation energy [J/mol]
T = 873                           # Temperature [K]
R = 8.1345                        # Ideal gas constant [J/Kmol]
Ndpart = 10**(-8)                 # Initial concentration of He-4 

# Grain dimensions
rmin = 0                          # Minimum radius [m]
rmax = np.copy(a)                 # Radius [m]
dr = 10**-6                       # Distance step [m]
r = np.arange(rmin, rmax,dr)      # Radius array [m]
Nd = np.zeros(len(r))             # Array of concentration [m]
Nd.fill(Ndpart)

# Calculate diffusivity to evalue stability to find time step.
D = D0a2*np.exp(-Ea/(R*T))        # Diffusivity [m^2/s]
dttheory = (dr**2)/(2*D*a**2)     # Max. time step for stability [s]
dt = 2*10**2                       # Actual time step [s]
if dttheory<=dt:                  # Check if dt is too small and possibly warn
    print('Unstable code. Time step too large')
    print('Your time step (dt) is:' + str(dt) + '. It should be less than:' + \
    str(dttheory))

# Concentrations, decay parameters, and molar masses
U_conc_ppm = 0.25                 # Uranium conc. [ppm] 
Th_conc_ppm = 1.                  # Thorium conc. [ppm]
Sm_conc_ppm = 4.                  # Samarium conc. [ppm]


lamb238 = 1.55*10**(-10)          # U-238 decay const. [1/s]
lamb235 = 9.85*10**(-10)          # U-235 decay const. [1/s]
lamb232 = 4.95*10**(-11)          # Th-232 decay const. [1/s]
lamb147 = 6.53*10**(-12)          # Sm-232 decay const. [1/s]

m238 = 238.                       # Molar mass U-238
m235 = 235.                       # Molar mass U-235
m232 = 232.                       # Molar mass Th-232
m147 = 147.                       # Molar mass Sm-147

U_conc = U_conc_ppm*10**-6/m238   # U conc. [mol/g]
Th_conc = Th_conc_ppm*10**-6/m235 # Th conc. [mol/g]
Sm_conc = Sm_conc_ppm*10**-6/m147 # Sm conc. [mol/g]

"""
Loop
"""   
total_time = 2*10**6              # Time for diffusion to take place (s)
pltint = 800                      # Number of loops between plots
time = np.arange(0,total_time,dt) # Time array, t         
dNdr = np.zeros(len(Nd)-1)        # Empty flux array for 2/r*dN/dr term
d2Ndr2 = np.zeros(len(Nd)-1)      # Empty flux array for d^2N/dr^2 term
q = np.zeros(len(Nd))             # Empty diff array for d^2N/dr^2 term
Ndlen = len(Nd)                   # Length of Nd just to save time later
counter = 0                       # Count which loop we're on

for i in np.array(time):
    
    # First find flux for 2/r dN/dr term. Also find gradient for first term.
    # Then find gradient for d^2N/dr^2. Gradient also accounts for first term.
    dNdr[1:] = (Nd[2:Ndlen]-\
    Nd[0:Ndlen-2])/(2*dr)
    dNdr[0]=(dNdr[1]-dNdr[0])/2*dr
    # d2Ndr2 = np.gradient(Nd)[0:Ndlen-1]/(dr**2)
    q[1:] = np.diff(Nd)/dr
    d2Ndr2 = np.diff(q)/dr    
    
    # Calculate change in concentration over time, U weighted for U-235 in 2nd
    # term. All terms weighted by no. of alphas produced in decay.   
    dNdt = D*a**2*(d2Ndr2+(2/r[1:Ndlen])*dNdr) + \
    8*lamb238*U_conc*np.exp(lamb238*dt) + \
    7*lamb235*(U_conc/137.88*dt)*np.exp(lamb235*dt) + \
    6*lamb232*Th_conc*np.exp(lamb232*dt) + \
    lamb147*Sm_conc*np.exp(lamb147*dt)   
    
    Nd[:(Ndlen-1)] += dNdt*dt
    Nd[Ndlen-1] = 0
    
    # For every nth plot, plot. Determined by pltint
    counter += 1
    if counter % pltint==0:
        
        # Create figure
        fig = plt.figure(1)
        plt.clf()

        ax = fig.add_subplot(111, projection='3d')
        
        # Converting to spherical coordinates
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        # Coords. for outer sphere marking surface of the grain
        x = (rmax) * np.outer(np.cos(u), np.sin(v))
        y = (rmax) * np.outer(np.sin(u), np.sin(v))
        z = (rmax) * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Using to convert from m to mm in plotting
        scaling_factor = 1000
        ax.plot_surface(x*scaling_factor, y*scaling_factor, z*scaling_factor\
        ,rstride=4, cstride=4, color='b', alpha=0.1)
        
        # Finding areas where the concentration is 1/2 of the initial conc.
        contour_index_range = np.where(Ndpart- Nd<Ndpart*0.9)
        

        # Error will arise if the inner circle is 0 because amax will fail
        try:
            contour_index = np.amax(contour_index_range)
            x = rmax*(r[contour_index-1]/rmax) * np.outer(np.cos(u), np.sin(v))
            y = rmax*(r[contour_index-1]/rmax) * np.outer(np.sin(u), np.sin(v))
            z = rmax*(r[contour_index-1]/rmax) * np.outer(np.ones(np.size(u)),np.cos(v))
            ax.plot_surface(x*scaling_factor, y*scaling_factor,\
            z*scaling_factor,  rstride=4, cstride=4, color='hotpink',alpha=1)
        except ValueError:
            pass
        
        # Setting plotting details
        titlefont = {'fontname':'Verdana'}
#        font = {'fontname':'Verdana',
#                'color':'black',
#                'weight':'normal',
#                'size':12}
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        ax.set_xlabel('x dimension (m)',**titlefont)
        ax.set_ylabel('y dimension (m)',**titlefont)
        ax.set_zlabel('z dimension (m)',**titlefont)
        ax.set_zlim(-a*scaling_factor, a*scaling_factor)
        time_string = str(round(i/(31.536*10**6),1))
        title = plt.title(' Magnetite Diffusion Example \n'+
        'Time elapsed: ' + time_string + ' yrs \n \n',**titlefont)   
        

        plt.pause(0.001)        
        plt.draw()  







    
    
    
    
   

