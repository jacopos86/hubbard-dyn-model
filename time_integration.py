import numpy as np
import expect_val
import Set_Hamilt
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
		
	return tdocc, Eoft
