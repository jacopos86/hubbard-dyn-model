import numpy as np

############################################
######  GS OCCUPATIONS #####################
############################################

def gs_occup( calc, eig, eigv ):
	
	#### INPUT1: calc -> calculation #######
	#### INPUT2: eig -> eigenvalues ########
	#### INPUT3: eigv -> states ############
	
	### OCCUPATION VECTOR ###########
	n = np.zeros(2)
	
	if calc == "Hubbard":
		
		####################################
		### 1- | 1up, 1dn > ------- ########
		### 2- | 1up, 2dn > ------- ########
		### 3- | 1dn, 2up > ------- ########
		### 4- | 2up, 2dn > ------- ########
		####################################
		
		nocc_op = np.zeros((4, 4, 2))
		nocc_op[0,0,0] = 2
		nocc_op[1,1,0] = 1
		nocc_op[2,2,0] = 1
		nocc_op[1,1,1] = 1
		nocc_op[2,2,1] = 1
		nocc_op[3,3,1] = 2
		
		### COMPUTE OCCUP EXPECTATION ######
		i = int(np.where(eig == min(eig))[0])
		gst = eigv[:,i]
		norm = 0.
		for i in range(len(gst)):
			norm = norm + gst[i] * np.conjugate(gst[i])
		norm = np.sqrt(norm)
		state = gst / norm
		
		for i in range(2): 
			v = np.matmul(nocc_op[:,:,i], state)
			n[i] = np.vdot(state, v).real
			
	elif calc == "Kohn-Sham" or calc == "Hartree-Fock":
		
		####################################
		######## 1- | 1up > ------- ########
		######## 2- | 2dn > ------- ########
		####################################
		
		s = eigv.shape
		### COMPUTE OCCUP EXPECTATION ######
		i = int(np.where(eig == min(eig))[0])
		gst = eigv[:,i]
		norm = 0.
		for i in range(len(gst)):
			norm = norm + gst[i] * np.conjugate(gst[i])
		norm = np.sqrt(norm)
		state = gst / norm
		
		n[:] = state[:] * np.conjugate(state[:]) * 2. + n[:]
		
	else:
		
		raise ValueError("calc variable is not set to any calculation")
		
	return n
	
############################################
##### TD OCCUPATIONS #######################
############################################

def td_occup( calc, wfc ):
	
	#### INPUT1: calc -> calculation #######
	#### INPUT2: wfc    ####################
	
	### OCCUPATION VECTOR ###########
	n = np.zeros(2)
	
	if calc == "Hubbard":
		
		####################################
		### 1- | 1up, 1dn > ------- ########
		### 2- | 1up, 2dn > ------- ########
		### 3- | 1dn, 2up > ------- ########
		### 4- | 2up, 2dn > ------- ########
		####################################
		
		nocc_op = np.zeros((4, 4, 2))
		nocc_op[0,0,0] = 2
		nocc_op[1,1,0] = 1
		nocc_op[2,2,0] = 1
		nocc_op[1,1,1] = 1
		nocc_op[2,2,1] = 1
		nocc_op[3,3,1] = 2
		
		for i in range(2): 
			v = np.matmul(nocc_op[:,:,i], wfc)
			n[i] = np.vdot(wfc, v).real
			
	elif calc == "Kohn-Sham" or calc == "Hartree-Fock":
		
		####################################
		######## 1- | 1up > ------- ########
		######## 2- | 2dn > ------- ########
		####################################
		
		n[:] = wfc[:] * np.conjugate(wfc[:]) * 2. + n[:]
		
	else:
		
		raise ValueError("calc variable is not set to any calculation")
		
	return n
	
############################################
##### GS ENERGY ############################
############################################

def H_expect_gs( calc, eig, U, t, v, n ):
	
	#### COMPUTE ENERGY EXPECT. ###########
	E = 0.
	
	if calc == "Hubbard":
		
		E = min(eig)
		
	elif calc == "Kohn-Sham":
		
		####### COMPUTE ENERGY ####################
		## E = 2 e + Ec + dvc dn / 2 - Ehx  #######
		## Ehx = U/2 (1 + (dn/2)^2)  ##############
		## Ec = -U^2/8 (1 - (n2-1)^2)^{5/2} #######
		###########################################
		
		dn = n[1] - n[0]
		Ehx = U / 2. * (1. + (dn / 2.) ** 2)
		Ec = -U ** 2 / 8. * (1. - (n[1] - 1.) ** 2) ** (5./2)
		vc = np.zeros(2)
		vc[:] = 5./8 * U ** 2  * (n[:] - 1.) * (1. - (n[:] - 1.) ** 2) ** (3./2)
		dvc = vc[1] - vc[0]
		E = 2. * min(eig) + Ec + dvc * dn / 2. - Ehx
		
	elif calc == "Hartree-Fock":
		
		############ COMPUTE ENERGY ###############
		## E = U/2(1 - (dn/2)^2) -2t sqrt(1+x^2) ##
		## x = d veff / (2*t)  ####################
		###########################################
		
		veff = np.zeros(2)
		veff[:] = v[:] + U * n[:] / 2.
		dveff = veff[1] - veff[0]
		x = dveff / (2. * t)
		dn = n[1] - n[0]
		E = U / 2. * (1. - (dn / 2.) ** 2) - 2. * t * np.sqrt(1. + x ** 2) 
		
	else:
		
		raise ValueError("calc variable is not set to any calculation")
		
	return E
	
###################################################
##### H expectation value -- td evolution #########
###################################################
def H_expect( calc, H, psi, U, t, v, n ):
	
	### compute td energy expectation #############
	Et = 0.
	
	if calc == "Hubbard":
		
		### Et = <psi|H|psi> ######################
		
		Hpsi = np.matmul(H, psi)
		Et= np.vdot(psi, Hpsi).real
		
	elif calc == "Kohn-Sham":
		
		####### COMPUTE ENERGY ####################
		## E = 2 e + Ec + dvc dn / 2 - Ehx  #######
		## Ehx = U/2 (1 + (dn/2)^2)  ##############
		## Ec = -U^2/8 (1 - (n2-1)^2)^{5/2} #######
		## e = < psi | Hks | psi >   ##############
		###########################################
		
		dn = n[1] - n[0]
		vc = np.zeros(2)
		Ehx= U / 2. * (1. + (dn / 2.) ** 2)
		Ec =-U ** 2 / 8. * (1. - (n[1] - 1.) ** 2) ** (5./2)
		vc[:] = 5./8 * U ** 2 * (n[:] - 1.) * (1. - (n[:] - 1.) ** 2) ** (3./2)
		dvc = vc[1] - vc[0]
		Hpsi = np.matmul(H, psi)
		e = np.vdot(psi, Hpsi)
		Et= 2.*e + Ec + dvc * dn / 2. - Ehx
		
	elif calc == "Hartree-Fock":
		
		############ COMPUTE ENERGY ###############
		## E = U/2(1 - (dn/2)^2) -2t sqrt(1+x^2) ##
		## x = d veff / (2*t)  ####################
		###########################################
		
		dn = n[1] - n[0]
		Ehx= U / 2. * (1. + (dn / 2.) ** 2)
		Hpsi = np.matmul(H, psi)
		e = np.vdot(psi, Hpsi)
		Et = 2. * e - Ehx
		
	else:
		
		raise ValueError("calc variable is not set to any calculation")
		
	return Et
