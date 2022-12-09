#Use pyuul to create 3d representations
from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils # the PyUUL utility module
import time,os,urllib # some standard python modules we are going to use
import torch
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from torch.autograd import Variable

#Convert to numpy array
def to_numpy(tensor, detach = False):
    if detach:
        return tensor.detach().to_dense().cpu().numpy()
    else:
        return tensor.to_dense().cpu().numpy()

#parse with pyuul
def parse_protein(pdb_path, hashing = 'Element_Hashing'):
    
    #Get coordinates and atomic names
    coords, atname = utils.parsePDB(pdb_path)
    #Get channel for each atom
    atoms_channel = utils.atomlistToChannels(atname, hashing = hashing)
    #Get atom radii
    radius = utils.atomlistToRadius(atname)
    return coords, atname, atoms_channel, radius

#represent coordinates as dataframe
def get_coord_df(coords, atname, atoms_channel, radius):
    #Create coordinate dataframe for visuallization
    coord_df = pd.DataFrame(coords.numpy()[0])
    coord_df.columns = ['x', 'y', 'z']
    coord_df.index = atname
    coord_df['channel'] = atoms_channel[0]
    coord_df['radius'] = radius[0]
    return coord_df

#Get any representation from pyuul
def create_3d_rep(coords, radius, atoms_channel, device, rep, resolution = 1):
    
    #Pass torch tensor to device
    coords = coords.to(device)
    radius = radius.to(device)
    atoms_channel = atoms_channel.to(device)
    
    #Voxel representation
    if rep == 'voxel':
        VoxelsObject = VolumeMaker.Voxels(device=device,sparse=True)
        VoxelRepresentation = VoxelsObject(coords, radius, atoms_channel, resolution = resolution)
        return VoxelRepresentation
    
    elif rep == 'surface':
        #Point cloud surface representation
        PointCloudSurfaceObject = VolumeMaker.PointCloudVolume(device=device)
        SurfacePoitCloud = PointCloudSurfaceObject(coords, radius)
        return SurfacePoitCloud
        
    elif rep == 'volume':
        #Point cloud volume representation
        PointCloudVolumeObject = VolumeMaker.PointCloudSurface(device=device)
        VolumePoitCloud = PointCloudVolumeObject(coords, radius)
        return VolumePoitCloud
    
    
    return SurfacePoitCloud, VolumePoitCloud, VoxelRepresentation

#Voxellize protein pdb
def rep_protein(protein_path, device, rep, hashing = 'Element_Hashing', resolution = 1):
    coords, atname, atoms_channel, radius = parse_protein(protein_path, hashing)
    tensor = create_3d_rep(coords, radius, atoms_channel, device, rep, resolution = resolution)
    return tensor

#Get mean dimensions 
def get_mean_dimensions(deeploc_af2_df):
    dims = []
    for i in range(len(deeploc_af2_df.index)):
        if i % 1000 == 0: print(i)

        #Generate voxels
        protein_path = deeploc_af2_df['PDB Path'].iloc[i]
        voxel = rep_protein(protein_path = protein_path, 
                            device = device,
                            rep = rep,
                            hashing = hashing,
                            resolution = resolution)
        dims.append(list(voxel.size()))
    #Get mean dimensions
    dims = np.array(dims).transpose()
    return dims.mean(1)


#Voxellize complexes with seperate channels for proteins and ligands
def voxellize_complex(protein_path, ligand_path, outpath, device, hashing={"C":0,"O":1,"N":2,"H":3}):
    
    if not os.path.isfile(outpath):
        
        prot_coords, prot_atname = utils.parsePDB(protein_path)
        lig_coords, lig_atname = utils.parsePDB(ligand_path)
        prot_atom_channel = utils.atomlistToChannels(prot_atname, hashing=hashing)
        lig_atom_channel = utils.atomlistToChannels(lig_atname, hashing=hashing)
        lig_atom_channel = torch.add(lig_atom_channel, 4)

        coords = torch.cat((prot_coords, lig_coords), 1)
        atname = [prot_atname[0] + lig_atname[0]]
        atom_channel = torch.tensor([list(prot_atom_channel[0]) + list(lig_atom_channel[0])]).to(device)
        radius = utils.atomlistToRadius(atname).to(device)

        volmaker = VolumeMaker.Voxels(device=device)
        voxellizedVolume = volmaker(coords.to(device), radius, atom_channel,resolution=1).to_dense()
        
        torch.save(voxellizedVolume, outpath)
        
#Construct affine matrix for 3d rotation and translation
def get_affine_matrix(thetax, thetay, thetaz, dx, dy, dz):
    translation_matrix = np.array([[
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1 ],
    ]])
    
    x_rot_matrix = np.array([[
        [1, 0, 0, 0],
        [0, np.cos(thetax), np.sin(thetax), 0],
        [0, -np.sin(thetax), np.cos(thetax), 0],
        [0, 0, 0, 1],
    ]])
    y_rot_matrix = np.array([[
        [np.cos(thetay), 0, -np.sin(thetay), 0],
        [0, 1, 0, 0],
        [np.sin(thetay), 0, np.cos(thetay), 0],
        [0, 0, 0, 1],
    ]])
    z_rot_matrix = np.array([[
        [np.cos(thetaz), np.sin(thetaz), 0, 0],
        [-np.sin(thetaz), np.cos(thetaz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]])
    
    affine_matrix = x_rot_matrix * y_rot_matrix * z_rot_matrix + translation_matrix
    
    return torch.tensor(affine_matrix[:, :3, :], dtype = torch.float32)

#Get random rotation and translation matrix
def get_random_affine():
    affine_matrix = get_affine_matrix(
                                      uniform(-1, 1) * np.pi/4,
                                      uniform(-1, 1) * np.pi/4,
                                      uniform(-1, 1) * np.pi/4,
                                      uniform(-.2, .2),
                                      uniform(-.2, .2),
                                      uniform(-.2, .2)
    )
    return affine_matrix

#Apply rotation with affine_grid
def apply_rotation(theta, x):
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)
    
    return theta

#Interpolate all voxels to be specific size
def resize_grid(x, device, channels, xdim, ydim, zdim):
    
    #Resize xdim, ydim, zdim
    x =  torch.nn.functional.interpolate(x, size=(xdim, ydim, zdim)) 
    
    #Resize channels
    N, x_channels = x.size()[0:2]
    if x_channels < channels:
        pad =  channels - x_channels
        padding = Variable(torch.zeros(N, pad, xdim, ydim, zdim)).to(device)
        x = torch.cat((x, padding), 1)
        
    return x