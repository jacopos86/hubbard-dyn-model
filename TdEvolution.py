import numpy as np
import os
import sys
import shutil
import expect_val
import Set_Hamilt
import scipy.linalg as la
from math import exp, sin, cos

#############################################################################################################
###### MODULE FOR THE TIME INTEGRATION OF THE SCHRODINGER EQUATION USING: ###################################
###### 4TH ORDER RUNGE KUTTA METHOD #########################################################################
###### dy / dt = F(t, y)  ###################################################################################
###### K1 = h F(tn, yn)   ###################################################################################
###### K2 = h F(tn+h/2, yn+K1/2) ############################################################################
###### K3 = h F(tn+h/2, yn+K2/2) ############################################################################
###### K4 = h F(tn+h, yn+K3) ################################################################################
###### yn+1 = yn + (K1 + 2K2 + 2K3 + K4)/6 ##################################################################
#############################################################################################################
def RK4( dt, T, InitState, calc, vt, t, U ):
	### INPUT1: dt --- time interval     ######
	### INPUT2: T --- final time         ######
	### INPUT3: InitState --- initial state ###
	### INPUT4: calc --- calculation     ######
	### INPUT5: vt --- external td potential ##
	### INPUT6: t --- hooping parameter #######
	### INPUT7: U --- Coulombic inter. ########
	
	###  number of steps #####
	N = int(T / dt)
	State = InitState
	tdevol = np.zeros((len(State), N+1), dtype=np.complex128)
	tdevol[:,0] = State[:]
	tdocc = np.zeros((2, N))
	Eoft = np.zeros(N)
	### external potential ###
	vext = np.zeros(2)
	###  run time evolution ##
	for i in range(N):
		
		### COMPUTE OCCUPATIONS ########
		n = expect_val.td_occup( calc, State )
		
		### ext. potential #############
		vext[:] = 0.
		vext[:] = vt[:,2*i]
		
		### calc type ##################
		if calc == "Hubbard":
			H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
		elif calc == "Kohn-Sham":
			H = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n )
		elif calc == "Hartree-Fock":
			H = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n )
		### energy #####################
		psi = State
		Et = expect_val.H_expect( calc, H, psi, U, t, vext, n )
		
		### ENERGY + OCCUPATIONS #######
		tdocc[:,i] = n[:]
		Eoft[i] = Et
		
		### F1 matrix ##################
		F1 = -1j * np.matmul(H, State)
		### K1 matrix ##################
		K1 = dt * F1
		State1 = State + K1 / 2.
		
		### COMPUTE OCCUPATIONS ########
		n1 = expect_val.td_occup(calc, State1)
		### ext. potential #############
		vext[:] = 0.
		vext[:] = vt[:,2*i+1]
		
		### calc type ##################
		if calc == "Hubbard":
			H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
		elif calc == "Kohn-Sham":
			H = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n1 )
		elif calc == "Hartree-Fock":
			H = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n1 )
		### F2 matrix ##################
		F2 = -1j * np.matmul(H, State1)
		### K2 matrix ##################
		K2 = dt * F2
		State2 = State + K2 / 2.
		
		### COMPUTE OCCUPATIONS ########
		n2 = expect_val.td_occup(calc, State2)
		
		### calc type ##################
		if calc == "Hubbard":
			H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
		elif calc == "Kohn-Sham":
			H = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n2 )
		elif calc == "Hartree-Fock":
			H = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n2 )
		### F3 matrix ##################
		F3 = -1j * np.matmul(H, State2)
		### K3 matrix ##################
		K3 = dt * F3
		State3 = State + K3
		
		### COMPUTE OCCUPATIONS ########
		n3 = expect_val.td_occup(calc, State3)
		### ext. potential #############
		vext[:] = 0.
		vext[:] = vt[:,2*i+2]
		
		### calc. type #################
		if calc == "Hubbard":
			H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
		elif calc == "Kohn-Sham":
			H = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n3 )
		elif calc == "Hartree-Fock":
			H = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n3 )
		### F4 matrix ##################
		F4 = -1j * np.matmul(H, State3)
		### K4 matrix ##################
		K4 = dt * F4
		
		### new state ##################
		State = State + (K1 + 2.*K2 + 2.*K3 + K4) / 6.
		tdevol[:,i+1] = State[:]
		#print(i)
		
	return tdocc, Eoft
	
##############################################
#### TIME DEPENDENT CALCULATION ##############
##############################################

#### EXTRACT DATA FROM FILE ##################

f = open("input2", "r")
lines = f.readlines()
vext = np.zeros(2)

for line in lines:
	l = line.strip()
	l = l.split()
	if len(l) > 0:
		### RUN TYPE ####
		if l[0] == "run":
			run = l[2]
		### CALC TYPE ###
		if l[0] == "calc":
			calc = l[2]
		### HOPPING #####
		if l[0] == "t":
			t = float(l[2])
		### U ###########
		if l[0] == "U":
			U = float(l[2])
		### maxiter #####
		if l[0] == "maxstep":
			maxstep = int(l[2])
		### beta ########
		if l[0] == "beta":
			beta = float(l[2])
		### toler #######
		if l[0] == "tol":
			tol = float(l[2])
		### ext. pot ####
		if l[0] == "dvext" and l[2] == "expr":
			dv_expr1 = l[4]
			dv_expr2 = l[6]
		### max. time ###
		if l[0] == "T":
			T = float(l[2])
		### dt ##########
		if l[0] == "dt":
			dt = float(l[2])
		### OUT DIRECTORY
		if l[0] == "outdir":
			outdir = l[2]
	else:
		pass
		
if run != "td":
	raise ValueError("run variable is not set to 'td'")
	sys.exit(1)
	
#####  SET OUTPUT DIRECTORY ######################
os.mkdir(outdir)
#####  DEFINE TIME LISTS #########################
tl1 = np.arange(0., T+dt, dt)
tl2 = np.arange(0., T+dt/2., dt/2.)

#####  EVALUATE EXPRESSION #######################
vt1 = np.zeros((2, len(tl1)))
for i in range(len(tl1)):
	x = tl1[i]
	vt1[0,i] = eval(dv_expr1)
	vt1[1,i] = eval(dv_expr2)
dvt1 = np.zeros(len(tl1))
dvt1[:] = vt1[0,:] - vt1[1,:]

vt2 = np.zeros((2, len(tl2)))
for i in range(len(tl2)):
	x = tl2[i]
	vt2[0,i] = eval(dv_expr1)
	vt2[1,i] = eval(dv_expr2)
dvt2 = np.zeros(len(tl2))
dvt2[:] = vt2[0,:] - vt2[1,:]

##### EXTERNAL GS POTENTIAL ######################
vext = np.zeros(2)
vext[0] = vt1[0,0]
vext[1] = vt1[1,0]

#####  START TIME DEPENDENT EVOLUTION ############

if calc == "Hubbard":
	
	### SET UP THE HAMILTONIAN ###########
	H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
	
	### COMPUTE GS #######################
	[eig, eigv] = la.eigh(H)
	eig = eig.real
	i = int(np.where(eig == min(eig))[0])
	InitState = eigv[:,i]
	
	### COMPUTE OCCUPATIONS ##############
	n = expect_val.gs_occup( calc, eig, eigv )
	
	### COMPUTE ENERGY ###################
	E = expect_val.H_expect_gs( calc, eig, U, t, vext, n )
	
	### START TIME DEPENDENT EVOLUTION ###
	### RUN RUNGE-KUTTA 4 ################
	tdocc, Eoft = RK4( dt, T, InitState, calc, vt2, t, U )
	
elif calc == "Kohn-Sham":
	
	it = 0
	###### INITIALIZE 1 ELECTRON PER SITE #########
	n = np.zeros(2)
	n[:] = 1.
	nlst = []
	nlst.append(n)
	###### START ITERATION CYCLE ##################
	while it < maxstep:
		n1 = nlst[-1]
		###### SET KS HAMILTONIAN #################
		Hks = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n1 )
		############# COMPUTE GS ##################
		[eig, eigv] = la.eigh(Hks)
		eig = eig.real
		############# DENSITY EXPECT ##############
		n2 = expect_val.gs_occup( calc, eig, eigv )
		dn = n2 - n1
		if abs(dn[0]) < tol and abs(dn[1]) < tol:
			nlst.append(n2)
			######### COMPUTE ENERGY ##############
			E = expect_val.H_expect_gs( calc, eig, U, t, vext, n2 )
			break
		else:
			n = beta * n2 + (1. - beta) * n1
			nlst.append(n)
		it = it + 1
	###### EXTRACT INITIAL STATE ##################
	n = nlst[-1]
	i = int(np.where(eig == min(eig))[0])
	gst = eigv[:,i]
	norm = 0.
	for i in range(len(gst)):
		norm = norm + gst[i] * np.conjugate(gst[i])
	norm = np.sqrt(norm)
	InitState = gst / norm
	
	###### START TIME DEPENDENT EVOLUTION #########
	tdocc, Eoft = RK4( dt, T, InitState, calc, vt2, t, U )
	
elif calc == "Hartree-Fock":
	
	it = 0
	####### INITIALIZE 1 ELECTRON PER SITE #########
	n = np.zeros(2)
	n[:] = 1.
	nlst = []
	nlst.append(n)
	####### START ITERATION CYCLE ##################
	while it < maxstep:
		n1 = nlst[-1]
		Hhf = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n1 )
		############## COMPUTE GS ##################
		[eig, eigv] = la.eigh(Hhf)
		eig = eig.real
		############ DENSITY EXPECT ################
		n2 = expect_val.gs_occup( calc, eig, eigv )
		dn = n2 - n1
		if abs(dn[0]) < tol and abs(dn[1]) < tol:
			nlst.append(n2)
			########## COMPUTE ENERGY ##############
			E = expect_val.H_expect_gs( calc, eig, U, t, vext, n2 )
			break
		else:
			n = beta * n2 + (1. - beta) * n1
			nlst.append(n)
		it = it + 1
	###### EXTRACT INITIAL STATE ###################
	n = nlst[-1]
	i = int(np.where(eig == min(eig))[0])
	gst = eigv[:,i]
	norm = 0.
	for i in range(len(gst)):
		norm = norm + gst[i] * np.conjugate(gst[i])
	norm = np.sqrt(norm)
	InitState = gst / norm
	
	######## START TIME DEPENDENT EVOLUTION ########
	tdocc, Eoft = RK4( dt, T, InitState, calc, vt2, t, U )
	
else:
	raise ValueError("calc variable is not set to any calculation")
	
### WRITE OUTPUT FILES #############################
### ENERGY + OCCUPATIONS ###########################
N = int(T / dt)

namef = outdir + "td_occupations.txt"
f = open(namef, 'w')
for i in range(N):
	f.write( "%.7f        " % tl1[i] + "%.10f        " % tdocc[0,i] + "%.10f        " % tdocc[1,i] + "%.10f" % (tdocc[0,i]+tdocc[1,i]) + "\n" )
f.close()

namef = outdir + "energy.txt"
f = open(namef, 'w')
for i in range(N):
	f.write( "%.7f        " % tl1[i] + "%.10f" % Eoft[i] + "\n" )
f.close()

shutil.copyfile("./input2", outdir + "input2")
