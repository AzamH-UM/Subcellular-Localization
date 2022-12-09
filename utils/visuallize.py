#Visuallize 3D representations
import py3Dmol
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz.dot import make_dot
import pandas as pd

#Plot Voxel representation
def plot_voxel(voxel_rep, channel):
    
    #Convert to numpy
    voxel_rep = voxel_rep[0][channel]
    
    #Cmap doesnt really do anything but I tried
    cmap = cm.autumn
    norm = Normalize(vmin=np.min(voxel_rep), vmax=np.max(voxel_rep))
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel_rep, facecolors = cmap(norm(voxel_rep)))
    return ax

def plot_voxel_scatter(voxel_rep, channel, cutoff = 0.1, size = 1):
    voxel_rep = voxel_rep[0][channel]
    #Get x, y, z dimensions
    X, Y, Z = [], [], []
    values = []
    len_x, len_y, len_z = voxel_rep.shape
    for x in range(len_x):
        for y in range(len_y):
            for z in range(len_z):
                X.append(x)
                Y.append(y)
                Z.append(z)
                values.append(voxel_rep[x][y][z])
    
    voxel_df = pd.DataFrame({'x':X, 'y':Y, 'z':Z, 'values':values})
    voxel_df = voxel_df[voxel_df['values'] > cutoff]
    
    fig = px.scatter_3d(voxel_df, x='x', y='y', z='z', opacity = 1, color = 'values')
    fig.update_traces(marker={'size': size})
    return fig
  
def plot_coord_df(coord_df, opacity = 1):
    fig = px.scatter_3d(coord_df, x='x', y='y', z='z', color = 'channel', size = 'radius', opacity = opacity)
    return fig

#Plot surface representation
def plot_cloud(cloud_rep, opacity = .1):
    cloud_df = pd.DataFrame(cloud_rep)
    cloud_df.columns = ['x', 'y', 'z']
    fig = px.scatter_3d(cloud_df, x='x', y='y', z='z', opacity = opacity)
    return fig

#Overlay two surface reps
def plot_both_clouds(surface_rep, volume_rep, opacity = .3):
    surface_df = pd.DataFrame(surface_rep)
    surface_df.columns = ['x', 'y', 'z']
    surface_df['hue'] = 'surface'
    volume_df = pd.DataFrame(volume_rep)
    volume_df.columns = ['x', 'y', 'z']
    volume_df['hue'] = 'volume'
    plot_df = pd.concat([surface_df, volume_df])
    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', opacity = opacity, color = 'hue')
    return fig

#Edit protein bfactor
def edit_bfactor(pdb_path, outpath, values):
    
    #Read pdb
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_path)
    
    #Set all bfactors of atoms
    for atom, val in zip(structure.get_atoms(), values):
        atom.set_bfactor(val) 

    #Write pdb
    io = PDBIO()
    io.set_structure(structure)
    io.save(outpath)
    return

#Visuallize pdb with py3dmol
#Use py3Dmol to visuallize structure
def visuallize_pdb(pdb_path, bfactor = False):
    #Create py3dmol viewer
    view = py3Dmol.view()
    
    #Atom selections
    chA = {'or':[{'resn':["PRO"], 'invert':True}, {'resn':["PRO"]}]} #not sure why PRO0 chain is not working
    lig = {'or':[{'resn':'3MO'}, {'resn':'LIG'}]}
    fad = {'resn':'FAD'}
    
    #Add PDB to viewer
    view.addModel(open(pdb_path,'r').read(),'pdb')
    if bfactor:
        view.setStyle(chA, {'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':1}}})
    else:   
        view.setStyle(chA, {'cartoon':{'color':'spectrum'}})
    #view.addSurface(py3Dmol.VDW,{'opacity':0.7})
    
    view.zoomTo(chA)
    
    return view

#Visuallize network
def vis_net(x, model):
    y = model(x)
    return make_dot(y.mean(), params = dict(model.named_parameters()))



