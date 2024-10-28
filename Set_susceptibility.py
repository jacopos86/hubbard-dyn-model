import numpy as np
import sys
import Set_Hamilt
import scipy.linalg as la
import expect_val

class Susceptibility():
	
	##### this class defines the system's susceptibility
	
	def __init__(self, T, dt):
		N = int(T / dt)+1
		self.chi_oft = np.zeros((2,2,N), dtype=np.complex128)
		self.Nt = N
		self.time = np.arange(0., T+dt, dt)
		
	#####  compute real time susceptibility ############
	
	def set_chi_oft(self, calc, U, t, vext, maxstep, tol, beta):
		
		#############################################################
		############# compute \chi(i1,i2,t) #########################
		#### = -i/hbar \sum_{a,b}' (fa-fb) e^{iw_ab t} <a|n(1)|b> ###
		#### <b|n(1')|a> \theta(t)          #########################
		#############################################################
		
		### compute GS depending on the calculation #################
		
		if calc == "Hubbard":
			
			### set up the Hamiltonian ###########
			H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
			
			### compute GS #######################
			[eig, eigv] = la.eigh(H)
			eig = eig.real
			Nst = len(eig)
			
			### set density operator #############
			nop = np.zeros((4, 4, 2))
			nop[0,0,0] = 2.
			nop[1,1,0] = 1.
			nop[2,2,0] = 1.
			nop[1,1,1] = 1.
			nop[2,2,1] = 1.
			nop[3,3,1] = 2.
			
			### set occupations ##################
			focc = np.zeros(Nst)
			focc[list(eig).index(min(eig))] = 1.
			
			wfa = np.zeros(Nst, dtype=np.complex128)
			wfb = np.zeros(Nst, dtype=np.complex128)
			chit = np.zeros(self.Nt, dtype=np.complex128)
			for i1 in range(2):
				for i2 in range(2):
					chit[:] = 0.
					for a in range(Nst):
						wfa[:] = eigv[:,a]
						for b in range(Nst):
							wfb[:] = eigv[:,b]
							r = np.matmul(nop[:,:,i1], wfb)
							nab = np.vdot(wfa, r)
							r = np.matmul(nop[:,:,i2], wfa)
							nba = np.vdot(wfb, r)
							wab = eig[a] - eig[b]
							chit[:] = -1j * (focc[a]-focc[b]) * np.exp(1j*wab*self.time[:]) * nab * nba + chit[:]
					self.chi_oft[i1,i2,:] = chit[:]
					
		elif calc == "Kohn-Sham":
			it = 0
			#### initialize electron density #######
			n = np.zeros(2)
			n[:] = 1.
			nlst = []
			nlst.append(n)
			#### start iteration cycle #############
			while it < maxstep:
				n1 = nlst[-1]
				Hks = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n1 )
				#### compute GS ####################
				[eig, eigv] = la.eigh(Hks)
				eig = eig.real
				#### compute electron density ######
				n2 = expect_val.gs_occup( calc, eig, eigv )
				dn = n2 - n1
				if abs(dn[0]) < tol and abs(dn[1]) < tol:
					nlst.append(n2)
					##### compute energy ###########
					E = expect_val.H_expect_gs( calc, eig, U, t, vext, n2 )
					break
				else:
					n = beta * n2 + (1. - beta) * n1
					nlst.append(n)
				it = it + 1
			n = nlst[-1]
			
			###### set up Hamiltonian ##############
			Hks = Set_Hamilt.KS_Hamilt_twosites( t, U, vext, n )
			###### compute GS electron density #####
			[eig, eigv] = la.eig(Hks)
			eig = eig.real
			Nst = len(eig)
			###### set occupations #################
			focc = np.zeros(Nst)
			focc[list(eig).index(min(eig))] = 2.
			###### set density operator ############
			nop = np.zeros((2, 2, 2))
			nop[0,0,0] = 1.
			nop[1,1,1] = 1.
			
			wfa = np.zeros(Nst, dtype=np.complex128)
			wfb = np.zeros(Nst, dtype=np.complex128)
			chit = np.zeros(self.Nt, dtype=np.complex128)
			for i1 in range(2):
				for i2 in range(2):
					chit[:] = 0.
					for a in range(Nst):
						wfa[:] = eigv[:,a]
						for b in range(Nst):
							wfb[:] = eigv[:,b]
							r = np.matmul(nop[:,:,i1],wfb)
							nab = np.vdot(wfa, r)
							r = np.matmul(nop[:,:,i2],wfa)
							nba = np.vdot(wfb, r)
							wab = eig[a] - eig[b]
							chit[:] = -1j * (focc[a]-focc[b]) * np.exp(1j*wab*self.time[:]) * nab * nba + chit[:]
					self.chi_oft[i1,i2,:] = chit[:]
					
		elif calc == "Hartree-Fock":
			it = 0
			#### initialize electron density #######
			n = np.zeros(2)
			n[:] = 1.
			nlst = []
			nlst.append(n)
			#### start iteration cycle #############
			while it < maxstep:
				n1 = nlst[-1]
				Hhf = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n1 )
				#### GS computation ################
				[eig, eigv] = la.eig(Hhf)
				eig = eig.real
				#### density expect. ###############
				n2 = expect_val.gs_occup( calc, eig, eigv )
				dn = n2 - n1
				if abs(dn[0]) < tol and abs(dn[1]) < tol:
					nlst.append(n2)
					#### compute energy ############
					E = expect_val.H_expect_gs( calc, eig, U, t, vext, n2 )
					break
				else:
					n = beta * n2 + (1. - beta) * n1
					nlst.append(n)
				it = it + 1
			n = nlst[-1]
			#### set up the Hamiltonian ############
			Hhf = Set_Hamilt.HF_Hamilt_twosites( t, U, vext, n )
			#### compute GS electron density #######
			[eig, eigv] = la.eig(Hhf)
			eig = eig.real
			Nst = len(eig)
			#### set occupations ###################
			focc = np.zeros(Nst)
			focc[list(eig).index(min(eig))] = 2.
			
			wfa = np.zeros(Nst, dtype=np.complex128)
			wfb = np.zeros(Nst, dtype=np.complex128)
			chit = np.zeros(self.Nt, dtype=np.complex128)
			for i1 in range(2):
				for i2 in range(2):
					chit[:] = 0.
					for a in range(Nst):
						wfa[:] = eigv[:,a]
						for b in range(Nst):
							wfb[:] = eigv[:,b]
							nab = np.vdot(wfa, wfb)
							nba = np.vdot(wfb, wfa)
							wab = eig[a] - eig[b]
							chit[:] = -1j * (focc[a]-focc[b]) * np.exp(1j*wab*self.time[:]) * nab * nba + chit[:]
					self.chi_oft[i1,i2,:] = chit[:]
					
		else:
			raise ValueError("calc variable is not set to any calculation")
			
	######  compute frequency dependent Fourier transform ############
	
	def set_chi_ofw(self):
		
		print("OK")
