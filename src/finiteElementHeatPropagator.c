/********************************************
 *
 * finiteElementHeatPropagator.c, in the C programming language, written for MATLAB MEX function generation
 * C script for heat propagation based on Monte Carlo input
 *
 * Copyright 2017, 2018 by Anders K. Hansen, DTU Fotonik
 *
 * This file is part of MCmatlab.
 *
 * MCmatlab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MCmatlab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MCmatlab.  If not, see <https://www.gnu.org/licenses/>.
 *
 ** COMPILING ON WINDOWS
 * Can be compiled using "mex COPTIMFLAGS='$COPTIMFLAGS -Ofast -fopenmp -std=c11 -Wall -pedantic' LDOPTIMFLAGS='$LDOPTIMFLAGS -Ofast -fopenmp -std=c11 -Wall -pedantic' -outdir helperfuncs\private .\src\finiteElementHeatPropagator.c ".\src\libut.lib""
 *
 * To get the MATLAB C compiler to work, try this:
 * 1. Go to MATLAB's addon manager and tell it to install the "Support for MinGW-w64 compiler"
 * 2. Type "mex -setup" in the MATLAB command window and ensure that MATLAB has set the C compiler to MinGW64
 * 3. mex should now be able to compile the code using the above command
 *
 ** COMPILING ON MAC
 * As of June 2017, the macOS compiler doesn't support libut (for ctrl+c 
 * breaking) or openmp (for multithreading).
 * This file can then be compiled with "mex COPTIMFLAGS='$COPTIMFLAGS -Ofast -std=c11 -Wall -pedantic' LDOPTIMFLAGS='$LDOPTIMFLAGS -Ofast -std=c11 -Wall -pedantic' -outdir helperfuncs/private ./src/finiteElementHeatPropagator.c"
 *
 * To get the MATLAB C compiler to work, try this:
 * 1. Install XCode from the App Store
 * 2. Type "mex -setup" in the MATLAB command window
 ********************************************/

#include <math.h>
#include "mex.h"
#ifdef _WIN32 // This is defined on both win32 and win64 systems. We use this preprocessor condition to avoid loading openmp or libut on, e.g., Mac
#include "omp.h"
extern bool utIsInterruptPending(); // Allows catching ctrl+c while executing the mex function
#endif

#define R 8.3144598 // Gas constant [J/mol/K]
#define CELSIUSZERO 273.15
#define T1 (*srcTemp)
#define T2 (*tgtTemp)

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, mxArray const *prhs[] ) {
    bool ctrlc_caught = false;           // Has a ctrl+c been passed from MATLAB?
    long nx,ny,nz,nM;
    
    mwSize const *dimPtr = mxGetDimensions(prhs[0]);
    nx = dimPtr[0];
    ny = dimPtr[1];
    nz = dimPtr[2];
    nM = mxGetM(mxGetField(prhs[2],0,"dTdtperdeltaT")); // Number of media in the simulation.
    
    
    double *Temp          = mxGetPr(prhs[0]); // Temp is an nx*ny*nz array of doubles
    plhs[0] = mxCreateNumericArray(3,dimPtr,mxDOUBLE_CLASS,mxREAL);
    double *outputTemp    = mxGetPr(plhs[0]); // outputTemp is an nx*ny*nz array of doubles
    long nt               = *mxGetPr(mxGetField(prhs[2],0,"steps")); // nt is an integer (number of time steps to perform)
    double dt             = *mxGetPr(mxGetField(prhs[2],0,"dt")); // dt is a double (duration of time step)
    unsigned char *M      = mxGetData(mxGetField(prhs[2],0,"M")); // M is a nx*ny*nz array of uint8 (unsigned char) containing values from 0..nM-1
    double *c             = mxGetPr(mxGetField(prhs[2],0,"dTdtperdeltaT")); // c is an nM*nM*3 array of doubles
    double *dTdt_abs      = mxGetPr(mxGetField(prhs[2],0,"dTdt_abs")); // dTdt_abs is an nx*ny*nz array of doubles
    double *A             = mxGetPr(mxGetField(prhs[2],0,"A")); // A is a nM array of doubles
    double *E             = mxGetPr(mxGetField(prhs[2],0,"E")); // E is a nM array of doubles
    bool lightsOn         = mxIsLogicalScalarTrue(mxGetField(prhs[2],0,"lightsOn"));
    long heatSinked       = *mxGetPr(mxGetField(prhs[2],0,"heatBoundaryType")); // 0: Insulating boundaries, 1: Constant-temperature boundaries
    
    double *Omega       = mxGetPr(prhs[1]); // Omega is an nx*ny*nz array of doubles if we are supposed to calculate damage, a single NaN element otherwise
    bool calcDamage     = !mxIsNaN(Omega[0]); // If the Omega input is just one NaN element then we shouldn't bother with thermal damage calculation
    plhs[1] = calcDamage? mxCreateNumericArray(3,dimPtr,mxDOUBLE_CLASS,mxREAL): mxCreateDoubleMatrix(1,1,mxREAL); // outputOmega is the same dimensions as Omega
    double *outputOmega = mxGetPr(plhs[1]);
    if(!calcDamage) outputOmega[0] = NAN;
    
    double *tempSensorCornerIdxs = mxGetPr(mxGetField(prhs[2],0,"tempSensorCornerIdxs"));
    long nSensors = mxGetM(mxGetField(prhs[2],0,"tempSensorCornerIdxs"));
    double *tempSensorInterpWeights = mxGetPr(mxGetField(prhs[2],0,"tempSensorInterpWeights"));
    
    plhs[2] = nSensors? mxCreateDoubleMatrix(nSensors,nt+1,mxREAL): mxCreateDoubleMatrix(0,0,mxREAL);
    double *sensorTemps = mxGetPr(plhs[2]);
    
    double *tempTemp = malloc(nx*ny*nz*sizeof(double)); // Temporary temperature matrix
    
    double **srcTemp = &Temp; // Pointer to the pointer to whichever temperature matrix is to be read from
    double **tgtTemp; // Pointer to the pointer to whichever temperature matrix is to be written to
    
    tgtTemp = ((nt+lightsOn)%2)? &outputTemp: &tempTemp; // Number of times the memory location has to change is nt if no light is on or nt+1 is lights are on. If that number is odd then we can start by writing to the output temperature matrix (pointed to by plhs[0]), otherwise we start by writing to the temporary temperature matrix
    
    for(long j=0;j<nSensors;j++) { // Interpolate to get the starting temperatures on the temperature sensors
        long idx = tempSensorCornerIdxs[j];
        double wx = tempSensorInterpWeights[j           ];
        double wy = tempSensorInterpWeights[j+  nSensors];
        double wz = tempSensorInterpWeights[j+2*nSensors];
        sensorTemps[j] = (1-wx)*(1-wy)*(1-wz)*Temp[idx     ] + (1-wx)*(1-wy)*wz*Temp[idx     +nx*ny] +
                         (1-wx)*   wy *(1-wz)*Temp[idx  +nx] + (1-wx)*   wy *wz*Temp[idx  +nx+nx*ny] +
                            wx *(1-wy)*(1-wz)*Temp[idx+1   ] +    wx *(1-wy)*wz*Temp[idx+1   +nx*ny] +
                            wx *   wy *(1-wz)*Temp[idx+1+nx] +    wx *   wy *wz*Temp[idx+1+nx+nx*ny];
    }
    
    #ifdef _WIN32
    bool useAllCPUs       = mxIsLogicalScalarTrue(mxGetField(prhs[2],0,"useAllCPUs"));
    #pragma omp parallel num_threads(useAllCPUs || omp_get_num_procs() == 1? omp_get_num_procs(): omp_get_num_procs()-1)
    #endif
    {
        long ix,iy,iz,of,jm1,j,jp1,jp2;
        long n,i;
        double dTdt,w; // Time derivative of temperature
		double b[nx>ny && nx>nz? nx: (ny>nz? ny: nz)];
        
		
		if(calcDamage) {
			// Initialize omega array
			#ifdef _WIN32
			#pragma omp for schedule(auto)
			#endif
			for(long idx=0;idx<nx*ny*nz;idx++) outputOmega[idx] = Omega[idx];
		}
		
		if(lightsOn) {
			// If lights are on, copy input temps to one of the modifiable memory locations, depositing half of the heat of this time step in the process
			#ifdef _WIN32
			#pragma omp for schedule(auto)
			#endif
			for(long idx=0;idx<nx*ny*nz;idx++) T2[idx] = T1[idx] + dt*dTdt_abs[idx]/2;
			#ifdef _WIN32
			#pragma omp master
			#endif
			{
				if(tgtTemp == &outputTemp) {
					tgtTemp = &tempTemp;
					srcTemp = &outputTemp;
				} else {
					tgtTemp = &outputTemp;
					srcTemp = &tempTemp;
				}
			}
			#ifdef _WIN32
			#pragma omp barrier
			#endif
		}
		
        for(n=0; n<nt; n++) {
            if(ctrlc_caught) break;
			
			if(n && lightsOn) {
				// If lights are on and this is not the first step, deposit half the heat of this time step
				#ifdef _WIN32
				#pragma omp for schedule(auto)
				#endif
				for(long idx=0;idx<nx*ny*nz;idx++) T1[idx] += dt*dTdt_abs[idx]/2;
			}
			
            // Explicit part of substep 1 out of 3
			#ifdef _WIN32
            #pragma omp for schedule(auto)
            #endif
            for(iz=0; iz<nz; iz++) {
                for(iy=0; iy<ny; iy++) {
                    for(ix=0; ix<nx; ix++) {
                        i = ix + iy*nx + iz*nx*ny;
						
						dTdt = 0;
                        if(!(heatSinked && (ix == 0 || ix == nx-1 || iy == 0 || iy == ny-1 || iz == 0 || iz == nz-1))) { // If not constant-temperature boundary voxel
                            if(ix != 0   ) dTdt += (T1[i-1]     - T1[i])*c[M[i] + M[i-1    ]*nM          ]/2; // Factor � is because the first substep splits the x heat flow over an explicit and an implicit part
                            if(ix != nx-1) dTdt += (T1[i+1]     - T1[i])*c[M[i] + M[i+1    ]*nM          ]/2; // Factor � is because the first substep splits the x heat flow over an explicit and an implicit part
                            if(iy != 0   ) dTdt += (T1[i-nx]    - T1[i])*c[M[i] + M[i-nx   ]*nM +   nM*nM];
                            if(iy != ny-1) dTdt += (T1[i+nx]    - T1[i])*c[M[i] + M[i+nx   ]*nM +   nM*nM];
                            if(iz != 0   ) dTdt += (T1[i-nx*ny] - T1[i])*c[M[i] + M[i-nx*ny]*nM + 2*nM*nM];
                            if(iz != nz-1) dTdt += (T1[i+nx*ny] - T1[i])*c[M[i] + M[i+nx*ny]*nM + 2*nM*nM];
                        }
                        T2[i] = T1[i] + dt*dTdt;
                    }
                }
            }
			
			// Implicit part of substep 1 out of 3
			#ifdef _WIN32
            #pragma omp for schedule(auto)
            #endif
			for(iz=heatSinked; iz<nz-heatSinked; iz++) {
                for(iy=heatSinked; iy<ny-heatSinked; iy++) {
					of = iy*nx + iz*ny*nx; // Index offset of this x-line
					
					// Thomson algorithm, sweeps up from 0 to nx-1 and then down from nx-1 to 0:
					// Algorithm is taken from https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
					
					// Forward sweep, indices 0 and 1:
					j   = of;
					jp1 = of+1;
					jp2 = of+2;
					b[0] = heatSinked? 1: 1 + dt/2*c[M[j] + nM*M[jp1]];
					w = -dt/2*c[M[jp1] + nM*M[j]]/b[0];
					b[1] = 1 + dt/2*c[M[jp1] + nM*M[jp2]] + dt/2*c[M[jp1] + nM*M[j]];
					if(!heatSinked) b[1] += w*dt/2*c[M[j] + nM*M[jp1]];
					T2[jp1] -= w*T2[j];
					// Forward sweep, indices 2 to nx-2:
					for(ix=2;ix<nx-1;ix++) {
						jm1 = of+ix-1;
						j   = of+ix;
						jp1 = of+ix+1;
						w = -dt/2*c[M[j] + nM*M[jm1]]/b[ix-1];
						b[ix] = 1 + dt/2*c[M[j] + nM*M[jm1]] + dt/2*c[M[j] + nM*M[jp1]];
						b[ix] += w*dt/2*c[M[jm1] + nM*M[j]];
						T2[j] -= w*T2[jm1];
					}
					// Forward sweep, index nx-1:
					jm1 = of+nx-2;
					j   = of+nx-1;
					if(heatSinked) {
						b[nx-1] = 1;
					} else {
						w = -dt/2*c[M[j] + nM*M[jm1]]/b[nx-2];
						b[nx-1] = 1 + dt/2*c[M[j] + nM*M[jm1]];
						b[nx-1] += w*dt/2*c[M[jm1] + nM*M[j]];
						T2[j] -= w*T2[jm1];
					}
					
					// Back sweep, index nx-1:
					T2[j] /= b[nx-1];
					// Back sweep, indices nx-2 to 1:
					for(ix=nx-2;ix>0;ix--) {
						j   = of+ix;
						jp1 = of+ix+1;
						T2[j] = (T2[j] + dt/2*c[M[j] + nM*M[jp1]]*T2[jp1])/b[ix];
					}
					// Back sweep, index 0:
					j   = of;
					jp1 = of+1;
					if(!heatSinked) {
						T2[j] = (T2[j] + dt/2*c[M[j] + nM*M[jp1]]*T2[jp1])/b[0];
					}
				}
			}
			
            // Explicit part of substep 2 out of 3
			#ifdef _WIN32
            #pragma omp for schedule(auto)
            #endif
            for(iz=0; iz<nz; iz++) {
                for(iy=0; iy<ny; iy++) {
                    for(ix=0; ix<nx; ix++) {
                        i = ix + iy*nx + iz*nx*ny;
						
						dTdt = 0;
                        if(!(heatSinked && (ix == 0 || ix == nx-1 || iy == 0 || iy == ny-1 || iz == 0 || iz == nz-1))) { // If not constant-temperature boundary voxel
                            if(iy != 0   ) dTdt -= (T1[i-nx]    - T1[i])*c[M[i] + M[i-nx   ]*nM +   nM*nM]/2;
                            if(iy != ny-1) dTdt -= (T1[i+nx]    - T1[i])*c[M[i] + M[i+nx   ]*nM +   nM*nM]/2;
                        }
                        T2[i] = T2[i] + dt*dTdt; // Note that we use T2 on the right hand side here
                    }
                }
            }
			
			// Implicit part of substep 2 out of 3
			#ifdef _WIN32
            #pragma omp for schedule(auto)
            #endif
			for(iz=heatSinked; iz<nz-heatSinked; iz++) {
                for(ix=heatSinked; ix<nx-heatSinked; ix++) {
					of = ix + iz*ny*nx; // Index offset of this y-line
					
					// Thomson algorithm, sweeps up from 0 to ny-1 and then down from ny-1 to 0:
					
					// Forward sweep, indices 0 and 1:
					j   = of;
					jp1 = of+nx;
					jp2 = of+nx*2;
					b[0] = heatSinked? 1: 1 + dt/2*c[nM*nM + M[j] + nM*M[jp1]];
					w = -dt/2*c[nM*nM + M[jp1] + nM*M[j]]/b[0];
					b[1] = 1 + dt/2*c[nM*nM + M[jp1] + nM*M[jp2]] + dt/2*c[nM*nM + M[jp1] + nM*M[j]];
					if(!heatSinked) b[1] += w*dt/2*c[nM*nM + M[j] + nM*M[jp1]];
					T2[jp1] -= w*T2[j];
					// Forward sweep, indices 2 to ny-2:
					for(iy=2;iy<ny-1;iy++) {
						jm1 = of+nx*(iy-1);
						j   = of+nx*iy;
						jp1 = of+nx*(iy+1);
						w = -dt/2*c[nM*nM + M[j] + nM*M[jm1]]/b[iy-1];
						b[iy] = 1 + dt/2*c[nM*nM + M[j] + nM*M[jm1]] + dt/2*c[nM*nM + M[j] + nM*M[jp1]];
						b[iy] += w*dt/2*c[nM*nM + M[jm1] + nM*M[j]];
						T2[j] -= w*T2[jm1];
					}
					// Forward sweep, index ny-1:
					jm1 = of+nx*(ny-2);
					j   = of+nx*(ny-1);
					if(heatSinked) {
						b[ny-1] = 1;
					} else {
						w = -dt/2*c[nM*nM + M[j] + nM*M[jm1]]/b[ny-2];
						b[ny-1] = 1 + dt/2*c[nM*nM + M[j] + nM*M[jm1]];
						b[ny-1] += w*dt/2*c[nM*nM + M[jm1] + nM*M[j]];
						T2[j] -= w*T2[jm1];
					}
					
					// Back sweep, index ny-1:
					T2[j] /= b[ny-1];
					// Back sweep, indices ny-2 to 1:
					for(iy=ny-2;iy>0;iy--) {
						j   = of+nx*iy;
						jp1 = of+nx*(iy+1);
						T2[j] = (T2[j] + dt/2*c[nM*nM + M[j] + nM*M[jp1]]*T2[jp1])/b[iy];
					}
					// Back sweep, index 0:
					j   = of;
					jp1 = of+nx;
					if(!heatSinked) {
						T2[j] = (T2[j] + dt/2*c[nM*nM + M[j] + nM*M[jp1]]*T2[jp1])/b[0];
					}
				}
			}
			
            // Explicit part of substep 3 out of 3
			#ifdef _WIN32
            #pragma omp for schedule(auto)
            #endif
            for(iz=0; iz<nz; iz++) {
                for(iy=0; iy<ny; iy++) {
                    for(ix=0; ix<nx; ix++) {
                        i = ix + iy*nx + iz*nx*ny;
						
						dTdt = 0;
                        if(!(heatSinked && (ix == 0 || ix == nx-1 || iy == 0 || iy == ny-1 || iz == 0 || iz == nz-1))) { // If not constant-temperature boundary voxel
                            if(iz != 0   ) dTdt -= (T1[i-nx*ny] - T1[i])*c[M[i] + M[i-nx*ny]*nM + 2*nM*nM]/2;
                            if(iz != nz-1) dTdt -= (T1[i+nx*ny] - T1[i])*c[M[i] + M[i+nx*ny]*nM + 2*nM*nM]/2;
                        }
                        T2[i] = T2[i] + dt*dTdt; // Note that we use T2 on the right hand side here
                    }
                }
            }
			
			// Implicit part of substep 3 out of 3
			#ifdef _WIN32
            #pragma omp for schedule(auto)
            #endif
			for(iy=heatSinked; iy<ny-heatSinked; iy++) {
                for(ix=heatSinked; ix<nx-heatSinked; ix++) {
					of = ix + iy*nx; // Index offset of this z-line
					
					// Thomson algorithm, sweeps up from 0 to nz-1 and then down from nz-1 to 0:
					
					// Forward sweep, indices 0 and 1:
					j   = of;
					jp1 = of+nx*ny;
					jp2 = of+nx*ny*2;
					b[0] = heatSinked? 1: 1 + dt/2*c[2*nM*nM + M[j] + nM*M[jp1]];
					w = -dt/2*c[2*nM*nM + M[jp1] + nM*M[j]]/b[0];
					b[1] = 1 + dt/2*c[2*nM*nM + M[jp1] + nM*M[jp2]] + dt/2*c[2*nM*nM + M[jp1] + nM*M[j]];
					if(!heatSinked) b[1] += w*dt/2*c[2*nM*nM + M[j] + nM*M[jp1]];
					T2[jp1] -= w*T2[j];
					// Forward sweep, indices 2 to nz-2:
					for(iz=2;iz<nz-1;iz++) {
						jm1 = of+nx*ny*(iz-1);
						j   = of+nx*ny* iz;
						jp1 = of+nx*ny*(iz+1);
						w = -dt/2*c[2*nM*nM + M[j] + nM*M[jm1]]/b[iz-1];
						b[iz] = 1 + dt/2*c[2*nM*nM + M[j] + nM*M[jm1]] + dt/2*c[2*nM*nM + M[j] + nM*M[jp1]];
						b[iz] += w*dt/2*c[2*nM*nM + M[jm1] + nM*M[j]];
						T2[j] -= w*T2[jm1];
					}
					// Forward sweep, index nz-1:
					jm1 = of+nx*ny*(nz-2);
					j   = of+nx*ny*(nz-1);
					if(heatSinked) {
						b[nz-1] = 1;
					} else {
						w = -dt/2*c[2*nM*nM + M[j] + nM*M[jm1]]/b[nz-2];
						b[nz-1] = 1 + dt/2*c[2*nM*nM + M[j] + nM*M[jm1]];
						b[nz-1] += w*dt/2*c[2*nM*nM + M[jm1] + nM*M[j]];
						T2[j] -= w*T2[jm1];
					}
					
					// Back sweep, index nz-1:
					T2[j] /= b[nz-1];
					// Back sweep, indices nz-2 to 1:
					for(iz=nz-2;iz>0;iz--) {
						j   = of+nx*ny*iz;
						jp1 = of+nx*ny*(iz+1);
						T2[j] = (T2[j] + dt/2*c[2*nM*nM + M[j] + nM*M[jp1]]*T2[jp1])/b[iz];
						if(lightsOn) T2[jp1] += dt*dTdt_abs[jp1]/2; // Deposition of half the heat of the step, strictly speaking not a part of the implicit step
						if(calcDamage) outputOmega[j] += dt*A[M[j]]*exp(-E[M[j]]/(R*((T2[j] + T1[j])/2 + CELSIUSZERO))); // Arrhenius damage integral evaluation, also not a part of the implicit step
					}
					// Back sweep, index 0:
					j   = of;
					jp1 = of+nx*ny;
					if(!heatSinked) {
						T2[j] = (T2[j] + dt/2*c[2*nM*nM + M[j] + nM*M[jp1]]*T2[jp1])/b[0];
					}
					if(lightsOn) T2[jp1] += dt*dTdt_abs[jp1]/2; // Deposition of half the heat of the step, strictly speaking not a part of the implicit step
					if(calcDamage) outputOmega[jp1] += dt*A[M[jp1]]*exp(-E[M[jp1]]/(R*((T2[jp1] + T1[jp1])/2 + CELSIUSZERO))); // Arrhenius damage integral evaluation, also not a part of the implicit step
					if(lightsOn) T2[j] += dt*dTdt_abs[j]/2; // Deposition of half the heat of the step, strictly speaking not a part of the implicit step
					if(calcDamage) outputOmega[j] += dt*A[M[j]]*exp(-E[M[j]]/(R*((T2[j] + T1[j])/2 + CELSIUSZERO))); // Arrhenius damage integral evaluation, also not a part of the implicit step
				}
			}

			#ifdef _WIN32
			#pragma omp master
			#endif
			{
				#ifdef _WIN32
				if(utIsInterruptPending()) {
					ctrlc_caught = true;
					printf("\nCtrl+C detected, stopping.\n");
				}
				#endif

				for(long j=0;j<nSensors;j++) { // Interpolate to get the new temperatures on the temperature sensors
					long idx = tempSensorCornerIdxs[j];
					double wx = tempSensorInterpWeights[j           ];
					double wy = tempSensorInterpWeights[j+  nSensors];
					double wz = tempSensorInterpWeights[j+2*nSensors];
					sensorTemps[j+(n+1)*nSensors] = (1-wx)*(1-wy)*(1-wz)*T2[idx     ] + (1-wx)*(1-wy)*wz*T2[idx     +nx*ny] +
													(1-wx)*   wy *(1-wz)*T2[idx  +nx] + (1-wx)*   wy *wz*T2[idx  +nx+nx*ny] +
													   wx *(1-wy)*(1-wz)*T2[idx+1   ] +    wx *(1-wy)*wz*T2[idx+1   +nx*ny] +
													   wx *   wy *(1-wz)*T2[idx+1+nx] +    wx *   wy *wz*T2[idx+1+nx+nx*ny];
				}

				if(tgtTemp == &outputTemp) {
					tgtTemp = &tempTemp;
					srcTemp = &outputTemp;
				} else {
					tgtTemp = &outputTemp;
					srcTemp = &tempTemp;
				}
			}
			#ifdef _WIN32
			#pragma omp barrier
			#endif
		}
    }
    
    free(tempTemp);
    return;
}
