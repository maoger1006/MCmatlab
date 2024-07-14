%% Description
% This example showcases GPU acceleration by executing on an Nvidia GPU
% through the CUDA architecture. If you have a CUDA-capable GPU you can
% launch MCmatlab with this acceleration by setting useGPU = true, as seen
% below. This flag exists both for the Monte Carlo light solver and the
% heat solver.
%
% The example has the same geometry as example 4, but with more time steps
% and fewer graphical updates in the heat solver so the CPU/GPU speed
% difference is more clear. First the MC step is run on CPU and then GPU -
% note the difference in number of photons simulated. Then the heat solver
% is run on CPU and then GPU - note the difference in time elapsed.
%
% On my Windows test PC with an Intel i5-6600 CPU and an Nvidia GeForce GTX
% 970 GPU, the speedup is a factor of 18x for the MC part and 7x for the
% heat part.
%
% Currently, only one GPU will be used even if you have multiple GPUs
% installed. If you are interested in using multiple GPUs in parallel,
% contact the MCmatlab developer Anders K. Hansen at
% anderskraghhansen@gmail.com

%% MCmatlab abbreviations
% G: Geometry, MC: Monte Carlo, FMC: Fluorescence Monte Carlo, HS: Heat
% simulation, M: Media array, FR: Fluence rate, FD: Fractional damage.
%
% There are also some optional abbreviations you can use when referencing
% object/variable names: LS = lightSource, LC = lightCollector, FPID =
% focalPlaneIntensityDistribution, AID = angularIntensityDistribution, NI =
% normalizedIrradiance, NFR = normalizedFluenceRate.
%
% For example, "model.MC.LS.FPID.radialDistr" is the same as
% "model.MC.lightSource.focalPlaneIntensityDistribution.radialDistr"

%% Geometry definition
MCmatlab.closeMCmatlabFigures();
model = MCmatlab.model;

model.G.nx                = 100; % Number of bins in the x direction
model.G.ny                = 100; % Number of bins in the y direction
model.G.nz                = 100; % Number of bins in the z direction
model.G.Lx                = 1; % [cm] x size of simulation cuboid
model.G.Ly                = 1; % [cm] y size of simulation cuboid
model.G.Lz                = 1.308+0.555; % [cm] z size of simulation cuboid

model.G.mediaPropertiesFunc = @mediaPropertiesFunc; % Media properties defined as a function at the end of this file
model.G.geomFunc          = @geometryDefinition; % Function to use for defining the distribution of media in the cuboid. Defined at the end of this m file.

model = plot(model,'G');

%% Monte Carlo simulation
model.MC.nPhotonsRequested  = 2e+08; % [min] Time duration of the simulation


model.MC.matchedInterfaces        = true; % Assumes all refractive indices are the same
model.MC.boundaryType             = 1; % 0: No escaping boundaries, 1: All cuboid boundaries are escaping, 2: Top cuboid boundary only is escaping, 3: Top and bottom boundaries are escaping, while the side boundaries are cyclic
model.MC.wavelength               = 660; % [nm] Excitation wavelength, used for determination of optical properties for excitation light


model.MC.lightSource.sourceType   = 4; % 0: Pencil beam, 1: Isotropically emitting line or point source, 2: Infinite plane wave, 3: Laguerre-Gaussian LG01 beam, 4: Radial-factorizable beam (e.g., a Gaussian beam), 5: X/Y factorizable beam (e.g., a rectangular LED emitter)
model.MC.lightSource.focalPlaneIntensityDistribution.radialDistr = 0; % Radial focal plane intensity distribution - 0: Top-hat, 1: Gaussian, Array: Custom. Doesn't need to be normalized.
model.MC.lightSource.focalPlaneIntensityDistribution.radialWidth = .03; % [cm] Radial focal plane 1/e^2 radius if top-hat or Gaussian or half-width of the full distribution if custom
model.MC.lightSource.angularIntensityDistribution.radialDistr = 0; % Radial angular intensity distribution - 0: Top-hat, 1: Gaussian, 2: Cosine (Lambertian), Array: Custom. Doesn't need to be normalized.
model.MC.lightSource.angularIntensityDistribution.radialWidth = 0; % [rad] Radial angular 1/e^2 half-angle if top-hat or Gaussian or half-angle of the full distribution if custom. For a diffraction limited Gaussian beam, this should be set to model.MC.wavelength*1e-9/(pi*model.MC.lightSource.focalPlaneIntensityDistribution.radialWidth*1e-2))
model.MC.lightSource.xFocus       = 0; % [cm] x position of focus
model.MC.lightSource.yFocus       = 0; % [cm] y position of focus
model.MC.lightSource.zFocus       = 0; % [cm] z position of focus
model.MC.lightSource.theta        = 0; % [rad] Polar angle of beam center axis
model.MC.lightSource.phi          = 0; % [rad] Azimuthal angle of beam center axis


% model.MC.useGPU                   = false; % (Default: false) Use CUDA acceleration for NVIDIA GPUs
% model = runMonteCarlo(model);
model.MC.useGPU                   = true; % (Default: false) Use CUDA acceleration for NVIDIA GPUs
model.MC.GPUdevice                = 0; % (Default: 0, the first GPU) The index of the GPU device to use for the simulation


n = 20; % Times to run
Run_num = NaN(1,n);

for iRun = 1:n
    model = runMonteCarlo(model);
    R1(iRun) = model.G.dx*model.G.dy*sum(model.MC.NI_zpos(:));
    R2(iRun) = model.G.dx*model.G.dy*sum(model.MC.NI_zneg(:));
    R3 (iRun)= model.G.dx*model.G.dz*sum(model.MC.NI_ypos(:));
    R4 (iRun)= model.G.dx*model.G.dz*sum(model.MC.NI_yneg(:));
    R5 (iRun)= model.G.dz*model.G.dy*sum(model.MC.NI_xpos(:));
    R6 (iRun)= model.G.dz*model.G.dy*sum(model.MC.NI_xneg(:));
    R = 1-(R1+R2+R3+R4+R5+R6)
end

model = plot(model,'MC');

writematrix(R1,'Results.xlsx','Sheet',2,'Range' , 'A4')

%% Geometry function(s) (see readme for details)
function M = geometryDefinition(X,Y,Z,parameters)
  % Blood vessel example:
  corneum_thick = 0.002;
  epidermis_thick = 0.025;
  papillary_thick = 0.01;
  upper_dermis = 0.008;
  reticular_dermis = 0.02;
  deep_dermis = 0.03;
  muscle_thick = 1*1.5;
  fat_thick = 0.055*1.5;
  boneradius  = 0.2;
  bonedepth = 0.6;
  M = ones(size(X)); % stratum corneum
  M(Z > corneum_thick) = 2; % epidermis
  M(Z > corneum_thick +   epidermis_thick) = 3; % papillary
  M(Z > corneum_thick + epidermis_thick + papillary_thick) = 4; %upper_dermis
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis ) = 5; % reticular_dermis
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + reticular_dermis ) = 9;
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + reticular_dermis + deep_dermis ) = 6;
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + reticular_dermis + deep_dermis + fat_thick)= 7;
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + reticular_dermis + deep_dermis + fat_thick + muscle_thick)= 6;                                                                                               
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + reticular_dermis + deep_dermis + 2*fat_thick + muscle_thick)= 9;
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + reticular_dermis + 2*deep_dermis + 2*fat_thick + muscle_thick)= 5;
  M(Z > corneum_thick + epidermis_thick + papillary_thick + upper_dermis + 2*reticular_dermis + 2*deep_dermis + 2*fat_thick + muscle_thick)= 4;
  M(Z > corneum_thick + epidermis_thick + papillary_thick + 2*upper_dermis + 2*reticular_dermis + 2*deep_dermis + 2*fat_thick + muscle_thick)= 3;
  M(Z > corneum_thick + epidermis_thick + 2*papillary_thick + 2*upper_dermis + 2*reticular_dermis + 2*deep_dermis + 2*fat_thick + muscle_thick)= 2;
  M(Z > corneum_thick + 2*epidermis_thick + 2*papillary_thick + 2*upper_dermis + 2*reticular_dermis + 2*deep_dermis + 2*fat_thick + muscle_thick)= 1;
  M(X.^2 + (Z - (corneum_thick+bonedepth)).^2 < boneradius^2) = 8;
end

%% Media Properties function (see readme for details)
function mediaProperties = mediaPropertiesFunc(parameters)
  mediaProperties = MCmatlab.mediumProperties;
  
  model.MC.wavelength = 660;
  Sa = 0.95;
  Sv = Sa - 0.1;
  mua_base = 7.84*(10^7)*(model.MC.wavelength )^(-3.255);
  Vmel =  0.1;

  j=1;
  mediaProperties(j).name  = 'stratum Corneum';
%   mediaProperties(j).mua = @func_mua1;
%   function mua = func_mua1(wavelength)
%     B = 0; % Blood content
%     S = 0; % Blood oxygen saturation
%     W = 0.05; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end    

%   mua_base = 7.84*(10^7)*(model.MC.wavelength )^(-3.255);
  
  mua_art = Sa*0.15 + (1-Sa)*1.64;
  mua_ven = Sv*0.15 + (1-Sv)*1.64;
  mua_w =  0.0036;
  
  Vart = 0;
  Vven = 0;
  Vw = 0.05;
  
  mediaProperties(j).mua = (Vart*mua_art + Vven*mua_ven + Vw*mua_w + (1-(Vart + Vven + Vw))* mua_base)/10;
  
  
  
  mediaProperties(j).mus = 2.562;  
  mediaProperties(j).g   = 0.9;
  mediaProperties(j).VHC = 3391*1.109e-3; % [J cm^-3 K^-1]
  mediaProperties(j).TC  = 0.37e-2; % [W cm^-1 K^-1]
  
  
%     mediaProperties(j).mus   = 44.5; % [cm^-1]
%     mediaProperties(j).g     = 0.8;
%     mediaProperties(j).n     = 1.3; % refractive indice

  j=2;
  mediaProperties(j).name  = 'Epidermis';
%   mediaProperties(j).mua = @func_mua2;
%   function mua = func_mua2(wavelength)
%     B = 0; % Blood content
%     S = 0.95; % Blood oxygen saturation
%     W = 0.2; % Water content
%     M = 0.1; % Melanin content  melanin only exist in epidermis
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mua_mel = 6.6*(10^10)*(model.MC.wavelength)^(-3.33);
  Vw = 0.2;
  mua_w =  0.0036;
  
  mediaProperties(j).mua = (Vmel*mua_mel +  Vw*mua_w + (1-(Vmel + Vw))* mua_base)/10;
%   mediaProperties(j).mua =  (mediaProperties(j).mua)/10;


%      mediaProperties(j).mus = @func_mus2;
%   function mus = func_mus2(wavelength)
%     aPrime = 66.7; % musPrime at 500 nm
%     fRay = 0.29; % Fraction of scattering due to Rayleigh scattering
%     bMie = 0.689; % Scattering power for Mie scattering
%     g = 0.9; % Scattering anisotropy
%     mus = calc_mus(wavelength,aPrime,fRay,bMie,g); % Jacques "Optical properties of biological tissues: a review" eq. 2
%   end

  mediaProperties(j).mus = 2.562;
  mediaProperties(j).g   = 0.9;
  mediaProperties(j).VHC = 3391*1.109e-3; % [J cm^-3 K^-1]
  mediaProperties(j).TC  = 0.37e-2; % [W cm^-1 K^-1]
  

  j=3;
  mediaProperties(j).name = 'papillary dermis';
%   mediaProperties(j).mua = @func_mua3;
%   function mua = func_mua3(wavelength)
%     B = 0.04; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.5; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end


  mua_art = Sa*0.15 + (1-Sa)*1.64;
  mua_ven = Sv*0.15 + (1-Sv)*1.64;
  mua_w =  0.0036;
  
  Vart = 0.04;
  Vven = 0.02;
  Vw = 0.5;
  
 mediaProperties(j).mua = (Vart*mua_art + Vven*mua_ven + Vw*mua_w + (1-(Vart + Vven + Vw))* mua_base)/10;
 

%   function mus = func_mus3(wavelength)
%     aPrime = 66.7; % musPrime at 500 nm
%     fRay = 0.29; % Fraction of scattering due to Rayleigh scattering
%     bMie = 0.689; % Scattering power for Mie scattering
%     g = 0.9; % Scattering anisotropy
%     mus = calc_mus(wavelength,aPrime,fRay,bMie,g); % Jacques "Optical properties of biological tissues: a review" eq. 2
%   end

  mediaProperties(j).mus = 2.562 
  mediaProperties(j).g   = 0.9;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;
  
  j=4;
  mediaProperties(j).name = 'upper blodd net dermis';
%   mediaProperties(j).mua = @func_mua4;
%   function mua = func_mua4(wavelength)
%     B = 0.3; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.6; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mua_art = Sa*0.15 + (1-Sa)*1.64;
  mua_ven = Sv*0.15 + (1-Sv)*1.64;
  mua_w =  0.0036;
  
  Vart = 0.3;
  Vven = 0.15;
  Vw = 0.6;
  
  mediaProperties(j).mua = (Vart*mua_art + Vven*mua_ven + Vw*mua_w + (1-(Vart + Vven + Vw))* mua_base)/10;
  
  
  mediaProperties(j).mus = 2.562 
  mediaProperties(j).g   = 0.9;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;
  
  j=5;
  mediaProperties(j).name = 'reticular dermis';
%   mediaProperties(j).mua = @func_mua5;
%   function mua = func_mua5(wavelength)
%     B = 0.04; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.7; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mua_art = Sa*0.15 + (1-Sa)*1.64;
  mua_ven = Sv*0.15 + (1-Sv)*1.64;
  mua_w =  0.0036;
  
  Vart = 0.04;
  Vven = 0.02;
  Vw = 0.7;
  
 mediaProperties(j).mua = (Vart*mua_art + Vven*mua_ven + Vw*mua_w + (1-(Vart + Vven + Vw))* mua_base)/10;
  
 
  mediaProperties(j).mus = 2.562 
  mediaProperties(j).g   = 0.9;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;
  
  j=6;
  mediaProperties(j).name = 'fat';
%   mediaProperties(j).mua = @func_mua6;
%   function mua = func_mua6(wavelength)
%     B = 0.0417; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.15; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mediaProperties(j).mua = 0.00104;
  mediaProperties(j).mus = 0.620 ;
  mediaProperties(j).g   = 0.8;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;
  
  j=7;
  mediaProperties(j).name = 'muscle';
%   mediaProperties(j).mua = @func_mua7;
%   function mua = func_mua7(wavelength)
%     B = 0.0417; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.65; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mediaProperties(j).mua = 0.00816;
  mediaProperties(j).mus = 0.861;  % Jacques "Optical properties of biological tissues: a review" eq. 2
  mediaProperties(j).g   = 0.5;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;
  
  
  j=8;
  mediaProperties(j).name = 'bone';
%   mediaProperties(j).mua = @func_mua8;
%   function mua = func_mua8(wavelength)
%     B = 0.0417; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.65; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mediaProperties(j).mua = 0.00351;
  mediaProperties(j).mus = 3.445;  % Jacques "Optical properties of biological tissues: a review" eq. 2
  mediaProperties(j).g   = 0.92;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;

 j=9;
  mediaProperties(j).name = 'deep blood net dermis';
%   mediaProperties(j).mua = @func_mua8;
%   function mua = func_mua8(wavelength)
%     B = 0.0417; % Blood content
%     S = 0.67; % Blood oxygen saturation
%     W = 0.65; % Water content
%     M = 0; % Melanin content
%     F = 0; % Fat content
%     mua = calc_mua(wavelength,S,B,W,F,M); % Jacques "Optical properties of biological tissues: a review" eq. 12
%   end

  mua_art = Sa*0.15 + (1-Sa)*1.64;
  mua_ven = Sv*0.15 + (1-Sv)*1.64;
  mua_w =  0.0036;
  
  Vart = 0.1;
  Vven = 0.05;
  Vw = 0.7;
  
  mediaProperties(j).mua = (Vart*mua_art + Vven*mua_ven + Vw*mua_w + (1-(Vart + Vven + Vw))* mua_base)/10;
  
  
  
  mediaProperties(j).mus = 2.562;  % Jacques "Optical properties of biological tissues: a review" eq. 2
  mediaProperties(j).g   = 0.92;   % Scattering anisotropy
  mediaProperties(j).n   = 1.3;
end