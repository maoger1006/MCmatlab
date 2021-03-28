classdef geometry
    %GEOMETRY This class contains all properties related to the geometry of
    %an MCmatlab.model.
    %   This class defines the properties of a geometry to be used in a
    %   monteCarloSimulation, fluorescenceMonteCarloSimulation or
    %   heatSimulation.
    
    properties
        silentMode logical = false              % Disables command window text and progress indication
        nx = NaN                                % Number of bins in the x direction
        ny = NaN                                % Number of bins in the y direction
        nz = NaN                                % Number of bins in the z direction
        Lx = NaN                                % [cm] x size of simulation cuboid
        Ly = NaN                                % [cm] y size of simulation cuboid
        Lz = NaN                                % [cm] z size of simulation cuboid
        mediaPropertiesFunc function_handle     % Media properties defined as a function at the end of the model file
        mediaPropParams cell = {}               % Cell array containing any additional parameters to be passed to the getMediaProperties function
        geomFunc function_handle                % Function to use for defining the distribution of media in the cuboid. Defined at the end of the model file.
        geomFuncParams cell = {}                % Cell array containing any additional parameters to pass into the geometry function, such as media depths, inhomogeneity positions, radii etc.
    end
    
    properties (Dependent)
        dx
        dy
        dz
        x
        y
        z
        M_raw uint8
    end
    
    methods
        function obj = geometry()
            %GEOMETRY Construct an instance of this class
            
        end
        
        function value = get.dx(obj)
            value = obj.Lx/obj.nx; % [cm] size of x bins
        end
        function value = get.dy(obj)
            value = obj.Ly/obj.ny; % [cm] size of y bins
        end
        function value = get.dz(obj)
            value = obj.Lz/obj.nz; % [cm] size of z bins
        end
        
        function value = get.x(obj)
            value = ((0:obj.nx-1)-(obj.nx-1)/2)*obj.dx; % [cm] x position of centers of voxels
        end
        function value = get.y(obj)
            value = ((0:obj.ny-1)-(obj.ny-1)/2)*obj.dy; % [cm] y position of centers of voxels
        end
        function value = get.z(obj)
            value = ((0:obj.nz-1)+1/2)*obj.dz; % [cm] z position of centers of voxels
        end
        
        function value = get.M_raw(obj)
            [X,Y,Z] = ndgrid(single(obj.x),single(obj.y),single(obj.z)); % The single data type is used to conserve memory
            value = uint8(obj.geomFunc(X,Y,Z,obj.geomFuncParams));
        end

    end
end

