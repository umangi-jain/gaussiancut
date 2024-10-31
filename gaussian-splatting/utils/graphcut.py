import numpy as np
import torch
import sys
sys.path.append('..')
import maxflow
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import copy
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

def load_gc_args(parser):
    # distance sigma for position distance between neighbouring gaussians
    parser.add_argument('--sig_pos_neigh', type=float, default=1)
    # distance sigma for color distance between neighbouring gaussians
    parser.add_argument('--sig_col_neigh', type=float, default=1)
    # distance sigma for position distance between non-terminal and terminal clusters
    parser.add_argument('--sig_pos_term', type=float, default=0.1)
    # weight for color distance between terminal clusters and non-terminal nodes
    parser.add_argument('--sig_col_term', type=float, default=1)
    # weight for color distance between neighbouring gaussians
    parser.add_argument('--weight_color', type=float, default=0.5)
    # weight for position distance between neighbouring gaussians
    parser.add_argument('--weight_pos', type=float, default=0.5)
    # weight for user input (computed from coarse splatting)
    parser.add_argument('--user_weight_term', type=float, default=1)
    # weight for minimum distance between non-terminal nodes and terminal clusters
    parser.add_argument('--cluster_term', type=float, default=1)
    

def find_cluster(point, cluster):
    """
    Finds the closest cluster from a given point.
    
    Parameters:
        point (numpy.ndarray): The point for which the distance is calculated.
        cluster (list[numpy.ndarray]): The cluster of points.
    
    Returns:
        Int:Tuple index of the closest cluster.
    """
    
    min_dist = np.inf
    min_cluster_index = -1
    for i in range(len(cluster)):
        c = cluster[i]
        dist = ((point - c)**2).sum()
        if dist < min_dist:
            min_dist = dist
            min_cluster_index = i
    
    return min_cluster_index

def nodes_correlation(pos_1, color_1, pos_2, color_2, weight_pos, weight_col, sig_pos, sig_col):
    neighbour_weight = 0
    pos_diff_neighbour = np.linalg.norm(pos_1-pos_2)
    pos_dist = weight_pos * np.exp(-sig_pos *pos_diff_neighbour)
    neighbour_weight +=  pos_dist 
    color_diff_neighbour = np.linalg.norm(color_1 - color_2)
    color_dist = weight_col * np.exp(-sig_col * color_diff_neighbour)
    neighbour_weight +=  color_dist
    return neighbour_weight
    
def graphcut_segmentation(args, root_path, graphcutparams, weights_source, weights_sink, gaussians, gaussians_source, 
                          gaussians_sink):
    """
    Perform graph cut segmentation on the given input graph of gaussians.

    Args:
        args: The arguments for the graph cut segmentation.
        root_path: The root path for saving the output files.
        graphcutparams: The parameters for the graph cut algorithm.
        weights_source: The weights for the source nodes.
        weights_sink: The weights for the sink nodes.
        gaussians: The input Gaussians.
        gaussians_source: The source Gaussians (used to find clusters).
        gaussians_sink: The sink Gaussian data (used to find clusters).

    Returns:
        gaussians_copy: The segmented Gaussian foreground.
        remove: The indices of the removed Gaussians (used later for rendering).
    """
    
    identifier = args.identifier
    sig_pos_neigh = args.sig_pos_neigh
    sig_col_neigh = args.sig_col_neigh
    sig_pos_term = args.sig_pos_term
    sig_col_term = args.sig_col_term
    user_weight_term = args.user_weight_term
    cluster_term = args.cluster_term
    weight_pos = args.weight_pos
    weight_color = args.weight_color 
    num_edges = graphcutparams.num_edges 

    weights_source = weights_source.detach().cpu().numpy()
    weights_sink = weights_sink.detach().cpu().numpy()
    
    pos = gaussians._xyz.detach().cpu().numpy()
    num_gaussians = pos.shape[0]
    colors = gaussians._features_dc.detach().cpu().numpy()[:, 0, :]

    # print('shape of gaussian positions: ', pos.shape)
    gaussians_copy = copy.deepcopy(gaussians)

    source_gauss_xyz = gaussians_source._xyz.detach().cpu().numpy()
    source_gauss_color = gaussians_source._features_dc.detach().cpu().numpy()   
    source_color = np.mean(source_gauss_color, axis=0)[0]  
    kmeans = KMeans(n_clusters=graphcutparams.terminal_clusters_source, random_state=0, n_init="auto").fit(source_gauss_xyz)
    source_cluster_nodes = kmeans.cluster_centers_
    print('source clusters: ', gaussians_source._xyz.shape, source_cluster_nodes.shape)
    
    sink_gauss_xyz = gaussians_sink._xyz.detach().cpu().numpy()
    sink_gauss_color = gaussians_sink._features_dc.detach().cpu().numpy()
    sink_color = np.mean(sink_gauss_color, axis=0)[0]
    kmeans = KMeans(n_clusters=graphcutparams.terminal_clusters_sink, random_state=0, n_init="auto").fit(sink_gauss_xyz)
    sink_cluster_nodes = kmeans.cluster_centers_
    print('sink clusters: ', gaussians_sink._xyz.shape, sink_cluster_nodes.shape)

    tree_pos = KDTree(pos, leaf_size=graphcutparams.leaf_size, metric='euclidean')
    _, ind = tree_pos.query(pos, k=num_edges)  
    
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(num_gaussians)

    for node in tqdm(range(num_gaussians), desc="Processing Gaussians"):
        neighbours = ind[node]
        for neigh_index in range(num_edges):
            neighbour = neighbours[neigh_index]
            if node == neighbour:
                continue
            neighbour_weight = nodes_correlation(pos[node], colors[node], pos[neighbour], 
                                                 colors[neighbour], weight_pos, weight_color, 
                                                 sig_pos_neigh, sig_col_neigh)

            g.add_edge(node, neighbour, neighbour_weight, neighbour_weight)   

        w_g = weights_source[node][0]
        w_gb = weights_sink[node][0]
        if w_gb == 0 and w_g == 0:
            w_gb = 0.01
        cluster_source = find_cluster(pos[node], source_cluster_nodes)
        cluster_weight_source = nodes_correlation(pos[node], colors[node], source_cluster_nodes[cluster_source], 
                                                 source_color, weight_pos, weight_color, 
                                                 sig_pos_term, sig_col_term)
        cluster_sink = find_cluster(pos[node], sink_cluster_nodes)
        cluster_weight_sink = nodes_correlation(pos[node], colors[node], sink_cluster_nodes[cluster_sink], 
                                                 sink_color, weight_pos, weight_color, 
                                                 sig_pos_term, sig_col_term)
        source_weight = user_weight_term*w_g + cluster_term*cluster_weight_source
        sink_weight = user_weight_term*(w_gb) + cluster_term*cluster_weight_sink
        
        g.add_tedge(node, source_weight, sink_weight)

    flow = g.maxflow()

    print(f"Maximum flow: {flow}")
    comps = []
    for i in range(num_gaussians):
        comps.append(g.get_segment(i))
    cluster_counts = np.bincount(comps)
    print('number of components in each cluster:', cluster_counts)
    remove = np.zeros((num_gaussians))
    for i in range(num_gaussians):
        if comps[i] == 1:
            remove[i] = 1
    print(remove.shape)


    gaussians_copy._xyz = torch.from_numpy(np.delete(gaussians._xyz.detach().cpu().numpy(), np.where(remove)[0], axis=0))
    gaussians_copy._features_dc = torch.from_numpy(np.delete(gaussians._features_dc.detach().cpu().numpy(), np.where(remove)[0], axis=0))
    gaussians_copy._features_rest = torch.from_numpy(np.delete(gaussians._features_rest.detach().cpu().numpy(), np.where(remove)[0], axis=0))
    gaussians_copy._opacity = torch.from_numpy(np.delete(gaussians._opacity.detach().cpu().numpy(), np.where(remove)[0], axis=0))
    gaussians_copy._scaling = torch.from_numpy(np.delete(gaussians._scaling.detach().cpu().numpy(), np.where(remove)[0], axis=0))
    gaussians_copy._rotation = torch.from_numpy(np.delete(gaussians._rotation.detach().cpu().numpy(), np.where(remove)[0], axis=0))
    print(gaussians_copy._features_rest.shape)

    gaussians_copy.save_ply(root_path + '/graphcut_{}/gaussians_source.ply'.format(identifier))
    torch.save(torch.from_numpy(remove), root_path + '/graphcut_{}/remove_source.pt'.format(identifier))
    return gaussians_copy, remove