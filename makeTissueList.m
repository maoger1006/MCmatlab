function [T, tissueList] = makeTissueList(T,wavelength)
%
%   Returns the known tissue properties (optical, thermal and/or fluorescence) at the specified wavelength.
%   Defining the thermal or fluorescence properties is only necessary if you want to run simulations based on
%   those properties.
%   
%   For each tissue, tissueList must contain mua, mus and g;
%       mua is the absorption coefficient [cm^-1] and must be positive (not zero)
%       mus is the scattering coefficient [cm^-1] and must be positive (not zero)
%       g is the anisotropy factor and must satisfy -1 <= g <= 1
%   and it may contain the refractive index for simulating non-matched boundaries such as reflection and refraction;
%       n is the refractive index and can be from 1 to inf, where inf means the medium is a perfect reflector
%   and parameters for simulating thermal diffusion;
%       VHC is volumetric heat capacity [J/(cm^3*K)] and must be positive
%       TC is thermal conductivity [W/(cm*K)] and must be non-negative
%   and parameters to calculate thermal damage;
%       E is the Arrhenius activation energy [J/mol]
%       A is the Arrhenius pre-exponential factor [1/s]
%   and parameters for fluorescence properties;
%       Y is fluorescence power yield (watts of emitted fluorescence light per watt of absorbed pump light) [-]
%       sat is saturation excitation intensity [W/cm^2]

%   Requires
%       SpectralLIB.mat

%% Acknowledgement
%   This function was inspired by makeTissueList of the mcxyz program hosted at omlc.org
%   Many parameters, formulas and the spectralLIB library is from mcxyz and
%   other work by Steven Jacques and collaborators.

%% Load spectral library
load spectralLIB.mat
%   muadeoxy      701x1              5608  double
%   muamel        701x1              5608  double
%   muaoxy        701x1              5608  double
%   muawater      701x1              5608  double
%   musp          701x1              5608  double
%   nmLIB         701x1              5608  double
MU(:,1) = interp1(nmLIB,muaoxy,wavelength);
MU(:,2) = interp1(nmLIB,muadeoxy,wavelength);
MU(:,3) = interp1(nmLIB,muawater,wavelength);
MU(:,4) = interp1(nmLIB,muamel,wavelength);

%% Create tissueList

j=1;
tissueList(j).name  = 'air';
tissueList(j).mua   = 1e-8;
tissueList(j).mus   = 1e-8;
tissueList(j).g     = 1;
tissueList(j).VHC   = 1.2e-3;
tissueList(j).TC    = 0; % Real value is 2.6e-4, but we set it to zero to neglect the heat transport to air
tissueList(j).n     = 1;

j=2;
tissueList(j).name  = 'water';
tissueList(j).mua   = 0.001;
tissueList(j).mus   = 10;
tissueList(j).g     = 1.0;
tissueList(j).VHC   = 4.19;
tissueList(j).TC    = 5.8e-3;
tissueList(j).n     = 1.3;

j=3;
tissueList(j).name  = 'standard tissue';
tissueList(j).mua   = 1;
tissueList(j).mus   = 100;
tissueList(j).g     = 0.9;
tissueList(j).VHC = 3391*1.109e-3;
tissueList(j).TC = 0.37e-2;
tissueList(j).n     = 1.3;

j=4;
tissueList(j).name  = 'epidermis';
B = 0;
S = 0.75;
W = 0.75;
M = 0.03;
musp500 = 40;
fray    = 0.0;
bmie    = 1.0;
gg      = 0.90;
musp = musp500*(fray*(wavelength/500).^-4 + (1-fray)*(wavelength/500).^-bmie);
X = [B*S B*(1-S) W M]';
tissueList(j).mua = MU*X;
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC =3391*1.109e-3;
tissueList(j).TC =0.37e-2;
tissueList(j).n     = 1.3;

j=5;
tissueList(j).name = 'dermis';
B = 0.002;
S = 0.67;
W = 0.65;
M = 0;
musp500 = 42.4;
fray    = 0.62;
bmie    = 1.0;
gg      = 0.90;
musp = musp500*(fray*(wavelength/500).^-4 + (1-fray)*(wavelength/500).^-bmie);
X = [B*S B*(1-S) W M]';
tissueList(j).mua = MU*X;
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC =3391*1.109e-3;
tissueList(j).TC =0.37e-2;
tissueList(j).n     = 1.3;

j=6;
tissueList(j).name  = 'blood';
B       = 1.00;
S       = 0.75;
W       = 0.95;
M       = 0;
musp500 = 10;
fray    = 0.0;
bmie    = 1.0;
gg      = 0.90;
musp = musp500*(fray*(wavelength/500).^-4 + (1-fray)*(wavelength/500).^-bmie);
X = [B*S B*(1-S) W M]';
tissueList(j).mua = MU*X;
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC   = 3617*1.050e-3;
tissueList(j).TC    = 0.52e-2;
tissueList(j).E   = 422.5e3; % J/mol    PLACEHOLDER DATA ONLY
tissueList(j).A   = 7.6e66; % 1/s        PLACEHOLDER DATA ONLY
tissueList(j).n     = 1.3;

j=7;
tissueList(j).name  = 'vessel';
tissueList(j).mua   = 0.8;
tissueList(j).mus   = 230;
tissueList(j).g     = 0.9;
tissueList(j).VHC   = 4200*1.06e-3;
tissueList(j).TC    = 6.1e-3;
tissueList(j).n     = 1.3;

j=8;
tissueList(j).name = 'enamel';
tissueList(j).mua   = 0.1;
tissueList(j).mus   = 30;
tissueList(j).g     = 0.96;
tissueList(j).VHC   = 750*2.97e-3;
tissueList(j).TC    = 9e-3;

j=9;
tissueList(j).name  = 'dentin';
tissueList(j).mua   = 4; %in cm ^ -1, doi:10.1364/AO.34.001278
tissueList(j).mus   = 270; %range between 260-280
tissueList(j).g     = 0.93;
tissueList(j).VHC   = 1260*2.14e-3; % Volumetric Heat Capacity [J/(cm^3*K)]
tissueList(j).TC    = 6e-3; % Thermal Conductivity [W/(cm*K)]

j=10;
tissueList(j).name = 'hair';
B = 0;
S = 0.75;
W = 0.75;
M = 0.03;
gg = 0.9;
musp = 9.6e10./wavelength.^3; % from Bashkatov 2002, for a dark hair
X = 2*[B*S B*(1-S) W M]';
tissueList(j).mua = MU*X; %Here the hair is set to absorb twice as much as the epidermis
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC = 1530*1.3e-3; % Thermal data has been approximated using the data for horn, as horn and hair are both composed of keratin
tissueList(j).TC  = 6.3e-3;

j=11;
tissueList(j).name  = 'glassfiber';
tissueList(j).mua   = 0.0001;
tissueList(j).mus   = 0.6666;
tissueList(j).g     = 0;
tissueList(j).VHC   = 703*2.203e-3;
tissueList(j).TC    = 13.8e-3;

j=12;
tissueList(j).name  = 'patch';
tissueList(j).mua   = 1119;
tissueList(j).mus   = 15;
tissueList(j).g     = 0.8;
tissueList(j).VHC   = 5.363*1.048e-3;
tissueList(j).TC    = 4.6e-3;

j=13;
tissueList(j).name  = 'skull';
% ONLY PLACEHOLDER DATA!
B = 0.0005;
S = 0.75;
W = 0.35;
M = 0;
musp500 = 30;
fray    = 0.0;
bmie    = 1.0;
gg      = 0.90;
musp = musp500*(fray*(wavelength/500).^-4 + (1-fray)*(wavelength/500).^-bmie);
X = [B*S B*(1-S) W M]';
tissueList(j).mua = MU*X;
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC = 3391*1.109e-3;
tissueList(j).TC = 0.37e-2;

j=14;
tissueList(j).name = 'gray matter';
% ONLY PLACEHOLDER DATA!
B = 0.01;
S = 0.75;
W = 0.75;
M = 0;
musp500 = 20;
fray    = 0.2;
bmie    = 1.0;
gg      = 0.90;
musp = musp500*(fray*(wavelength/500).^-4 + (1-fray)*(wavelength/500).^-bmie);
X = [B*S B*(1-S) W M]';
tissueList(j).mua = MU*X;
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC = 3391*1.109e-3;
tissueList(j).TC = 0.37e-2;
tissueList(j).n     = 1.3;

j=15;
tissueList(j).name  = 'white matter';
% ONLY PLACEHOLDER DATA!
B = 0.01;
S = 0.75;
W = 0.75;
M = 0;
musp500 = 20;
fray    = 0.2;
bmie    = 1.0;
gg      = 0.90;
musp = musp500*(fray*(wavelength/500).^-4 + (1-fray)*(wavelength/500).^-bmie);
X = [B*S B*(1-S) W M]';
tissueList(j).mua = MU*X;
tissueList(j).mus = musp/(1-gg);
tissueList(j).g   = gg;
tissueList(j).VHC = 3391*1.109e-3;
tissueList(j).TC = 0.37e-2;
tissueList(j).n     = 1.3;

j=16;
tissueList(j).name  = 'test fluorescing tissue';
if(wavelength<500)
    tissueList(j).mua = 100;
    tissueList(j).mus = 100;
    tissueList(j).g   = 0.9;

    tissueList(j).Y   = 0.5;
    tissueList(j).sat = 500;
else
    tissueList(j).mua = 1;
    tissueList(j).mus = 100;
    tissueList(j).g   = 0.9;
end

j=17;
tissueList(j).name  = 'test fluorescence absorber';
if(wavelength<500)
    tissueList(j).mua = 1;
    tissueList(j).mus = 100;
    tissueList(j).g   = 0.9;
else
    tissueList(j).mua = 100;
    tissueList(j).mus = 100;
    tissueList(j).g   = 0.9;
end

j=18;
tissueList(j).name  = 'testscatterer';
tissueList(j).mua   = 0.0000001;
tissueList(j).mus   = 100;
tissueList(j).g     = 0;

j=19;
tissueList(j).name  = 'testabsorber';
tissueList(j).mua   = 10000000000;
tissueList(j).mus   = 1;
tissueList(j).g     = 0;

j=20;
tissueList(j).name  = 'reflector';
tissueList(j).mua   = 1;
tissueList(j).mus   = 1;
tissueList(j).g     = 0;
tissueList(j).n     = inf;

%% Trim tissueList down to use only the tissues included in the input matrix T, and reduce T accordingly
nT = length(unique(T)); % Number of different tissues in simulation
tissueMap = zeros(1,length(tissueList),'uint8');
tissueMap(unique(T)) = 1:nT;
tissueList = tissueList(unique(T)); % Reduced tissue list, containing only the used tissues
T = tissueMap(T); % Reduced tissue matrix, using only numbers from 1 up to the number of used tissues

%% Fill in fluorescence and Arrhenius parameter assumptions
% For all tissues for which the fluorescence power yield Y, Arrhenius
% activation energy E or Arrhenius pre-exponential factor A was not 
% specified, assume they are zero. Also, if the fluorescence saturation
% was not specified, assume it is infinite.
for j=1:length(tissueList)
    if(~isfield(tissueList,'Y') || isempty(tissueList(j).Y))
        tissueList(j).Y = 0;
    end
    if(~isfield(tissueList,'E') || isempty(tissueList(j).E))
        tissueList(j).E = 0;
    end
    if(~isfield(tissueList,'A') || isempty(tissueList(j).A))
        tissueList(j).A = 0;
    end
    if(~isfield(tissueList,'sat') || isempty(tissueList(j).sat))
        tissueList(j).sat = Inf;
    end
end

%% Throw an error if a variable doesn't conform to its required interval
for j=1:length(tissueList)
    if(tissueList(j).mua <= 0)
        error('tissue %s has mua <= 0',tissueList(j).name);
    elseif(tissueList(j).mus <= 0)
        error('tissue %s has mus <= 0',tissueList(j).name);
    elseif(abs(tissueList(j).g) > 1)
        error('tissue %s has abs(g) > 1',tissueList(j).name);
    elseif(tissueList(j).n < 1)
        error('tissue %s has n < 1',tissueList(j).name);
    elseif(tissueList(j).VHC <= 0)
        error('tissue %s has VHC <= 0',tissueList(j).name);
    elseif(tissueList(j).TC < 0)
        error('tissue %s has TC < 0',tissueList(j).name);
    elseif(tissueList(j).Y < 0)
        error('tissue %s has Y < 0',tissueList(j).name);
    elseif(tissueList(j).sat <= 0)
        error('tissue %s has sat <= 0',tissueList(j).name);
    elseif(tissueList(j).E < 0)
        error('tissue %s has E < 0',tissueList(j).name);
    elseif(tissueList(j).A < 0)
        error('tissue %s has A < 0',tissueList(j).name);
    end
end

end