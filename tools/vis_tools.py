import io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba

import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

def show_save_image(image, figsize=(4,3), save_path=None):
    """
    Show and save an image.
    Args:
        image: numpy array, the image to show.
        figsize: tuple, the size of the figure.
        save_path: str, the path to save the image. If None, the image will not be saved.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    # remove the axis and white space
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_images(images, titles, figsize=(18, 3), save_path=None, show=True):
    """
    Plot a list of images.
    Args:
        images: list, a list of images in numpy array format.
        titles: list, a list of titles for each image.
        figsize: tuple, the size of the figure.
        save_path: str, the path to save the image. If None, the image will not be saved.
        show: bool, whether show the image.
    """
    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(titles[i])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()


def get_img_from_fig(fig, dpi=180, pad_inches=0):
    """ A function which returns an image as numpy array from plt figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def batch_show(imgs, sub_titles=None, title=None, row_labels=None,
               col_labels=None, cmap='gray', vrange_mode='fixed',
               ret_fig=False, font_size=(20, 20, 20),
               font_type='Times New Roman', sub_size=(3, 3)):
    """ Show images. 
    Args:
        imgs: Supposed to be an 2-d list or tuple. Each element is an image in numpy.ndarray format.
        sub_titles: Titles of each subplot.
        title: The image overall title.
        cmap: When the image only has two dimension, or only select one band, the cmap used by
            matplotlib.pyplot. Default is gray.
        vrange_mode: When the input image is monochrome, whether use a cmap value range auto min-max,
            or use a fixed range from 0 to 255. Select from ('auto', 'fixed').
        ret_fig: Whether return the processed input image.
        font_size: tuple/list/int/float, the font sizes of row, column, and subtitle. If input type is
            int/float, set all font sizes the same.
        font_type: str, the font name of your desired font type.
    """
    if not (isinstance(imgs[0], list) or isinstance(imgs[0], tuple)):
        imgs = [imgs]
    if not (isinstance(font_size, list) or isinstance(font_size, tuple)):
        font_size = (font_size, font_size, font_size)
    rows = len(imgs)
    cols = max([len(i) for i in imgs])

    # plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(sub_size[0]*cols, sub_size[1]*rows), sharey=True)
    if rows == 1:
        axs = [axs]
    if cols == 1:
        axs = [[i] for i in axs]
    axs = np.array(axs)
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            img = imgs[i][j]
            if sub_titles is not None and len(sub_titles) > i and len(sub_titles[i]) > j:
                sub_title = sub_titles[i][j]
            else:
                sub_title = ''
            if len(img.shape) == 2 or img.shape[0] == 1 or img.shape[-1] == 1:
                if vrange_mode == 'fixed':
                    axs[i, j].imshow(img, cmap=cmap, vmin=0, vmax=255)
                else:
                    axs[i, j].imshow(img, cmap=cmap)
            else:
                axs[i, j].imshow(img)
            axs[i, j].set(xticks=[], yticks=[])
            if row_labels is not None and len(row_labels) > i:
                axs[i, j].set_ylabel(row_labels[i], fontsize=font_size[0], fontname=font_type)
            if col_labels is not None and len(col_labels) > j:
                axs[i, j].set_xlabel(col_labels[j], fontsize=font_size[1], fontname=font_type)
            if sub_title != '':
                axs[i, j].set_title(sub_title, fontsize=font_size[2], y=-0.15, fontname=font_type)

    for ax in axs.flat:
        ax.label_outer()

    if title is not None:
        fig.suptitle(title, fontsize=30)
    plt.tight_layout()

    if ret_fig:
        return fig

def draw_cube(ax, x, y, z, color):
    """
    Draw a cube with a lower corner at x,y,z with dimensions dx,dy,dz.
    Args:
        ax: matplotlib 3d axis
        x: int, x coordinate of the lower corner of the cube
        y: int, y coordinate of the lower corner of the cube
        z: int, z coordinate of the lower corner of the cube
        color: str, color of the cube
    """
    # Define the vertices that compose each face of the cube
    vertices = np.array([[x, y, z],
                         [x+1, y, z],
                         [x+1, y+1, z],
                         [x, y+1, z],
                         [x, y, z+1],
                         [x+1, y, z+1],
                         [x+1, y+1, z+1],
                         [x, y+1, z+1]])
    # Define the vertices that compose each face
    faces = [[vertices[0], vertices[1], vertices[5], vertices[4]],
             [vertices[7], vertices[6], vertices[2], vertices[3]],
             [vertices[0], vertices[3], vertices[7], vertices[4]],
             [vertices[1], vertices[2], vertices[6], vertices[5]],
             [vertices[7], vertices[4], vertices[5], vertices[6]],
             [vertices[0], vertices[3], vertices[2], vertices[1]]]
    # Create the 3D cube
    if color == 'white':
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.25, edgecolors=to_rgba('lightgrey',0.1), alpha=0.02))
    else:
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.25, edgecolors=to_rgba('lightgrey',0.1), alpha=0.2))


def vis_cubes_plt(voxel):
    """
    Plot a 3d grid of transparent cubes, color of each cube is determined by the polarity of the voxel.
    Args:
        voxel: (2, C, H, W), a voxel with 2 channels, positive and negative.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through voxel grid
    for t in range(voxel.shape[1]):
        for y in range(voxel.shape[2]):
            for x in range(voxel.shape[3]):
                if voxel[0, t, y, x] and voxel[1, t, y, x]:
                    # select a color between lightgreen and lightcoral
                    color = 'moccasin'
                elif voxel[0, t, y, x]:
                    color = 'lightgreen'
                elif voxel[1, t, y, x]:
                    color = 'lightcoral' #'lightblue' #
                else:
                    color = 'white'
                draw_cube(ax, t, y, x, color)

    # Labeling and formatting
    ax.set_xlabel('T')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_title("3D Grid of Transparent Cubes")

    # set the axes limits
    ax.set_xlim3d(0, voxel.shape[1])
    ax.set_ylim3d(0, voxel.shape[2])
    ax.set_zlim3d(0, voxel.shape[3])

    # make the x:y:z ratio 2,1,1
    ax.set_box_aspect((1.5, 1, 1))

    # set the resolution of the plot to 256x128x128
    fig.set_dpi(128)

    # change the color of the axis lines to lightgrey
    for ax in fig.axes:
        ax.xaxis.line.set_color('grey')
        ax.yaxis.line.set_color('grey')
        ax.zaxis.line.set_color('grey')

    plt.show()
    
def plot_3d_scatter_plt(voxel, dpi=150, title="3D Scatter Plot", save_path=None):
    """ Plot a 3d scatter plot for a voxel.
    Args:
        voxel_vis: (2, C, H, W), a voxel with 2 channels, positive and negative.
        dpi: int, the resolution of the plot.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # set plt font to Helvetica
    # plt.rcParams['font.family'] = 'Helvetica'

    # scatter plot for positive voxels
    pos_voxel = voxel[0]
    pos_voxel = np.argwhere(pos_voxel)
    pos_voxel = pos_voxel[:,[0,1,2]]
    t = pos_voxel[:,0]
    y = 260-pos_voxel[:,1]
    x = pos_voxel[:,2]
    ax.scatter(t, x, y, c='lightgreen', marker='o', alpha=0.4, s=3, edgecolors='none', label='Positive')

    # scatter plot for negative voxels
    neg_voxel = voxel[1]
    neg_voxel = np.argwhere(neg_voxel)
    neg_voxel = neg_voxel[:,[0,1,2]]
    t = neg_voxel[:,0]
    y = 260-neg_voxel[:,1]
    x = neg_voxel[:,2]
    ax.scatter(t, x, y, c='lightcoral', marker='o', alpha=0.4, s=3, edgecolors='none', label='Negative')

    # Add legend
    ax.legend(loc='upper right', fontsize=12)

    # Labeling and formatting
    ax.set_xlabel('T')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_title(title)

    # set the axes limits
    ax.set_xlim3d(0, voxel.shape[1])
    ax.set_ylim3d(0, voxel.shape[3])
    ax.set_zlim3d(0, voxel.shape[2])

    # make the x:y:z ratio 2,1,1
    ax.set_box_aspect((1.5, 1, 0.75))
    plt.tight_layout()

    # set the resolution of the plot to 256x128x128
    fig.set_dpi(dpi)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)

    plt.show()
    
def plot_raw_events_xyt_plotly(events, timespan=1500, out_path=None):
    """
    Plot a 3d scatter plot for a input raw events.
    Args:
        events: (N, 4) numpy array, each row is an event with format [t, x, y, p]
        timespan: int, the time span of the plot.
        out_path: str, the path to save the plot. If None, the plot will not be saved.
    """
    color = events[:,3]
    t = events[:,0]
    x = events[:,1]
    y = 260-events[:,2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=t,
        mode='markers',
        marker=dict(
            size=2,
            color=color,                # set color to an array/list of desired values
            # self define a discrete color sequence for 0 and 1 (0: lightgreen, 1: lightcoral)
            colorscale=[[0, 'lightgreen'], [1, 'lightcoral']],
            # colorscale='Viridis',   # choose a colorscale
            opacity=0.75
        )
    )])

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[0,346],),
            yaxis = dict(nticks=4, range=[0,260],),
            zaxis = dict(nticks=4, range=[0,timespan],),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10),
        )

    # label set to X, Y, T
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Time'))

    camera = {
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 1.6007038109391156, 'y': -1.2165995503012192, 'z': 0.8032016209220834},
        'projection': {'type': 'orthographic'},
        'up': {'x': -0.513018666931529, 'y': -0.8269438383730598, 'z': -0.23016458362794973},
    }
    fig.update_layout(scene_camera=camera)
    
    # update the aspectratio 
    fig.update_layout(scene_aspectratio={'x': 1, 'y': 0.75, 'z': 2})

    # save the figure
    if out_path is not None:
        pio.write_image(fig, out_path, width=1200, height=800, scale=4)

    ## Uncomment if you want to extract the inner parameters from a plotly figure
    # f2 = go.FigureWidget(fig)
    # f2
    
    # Show plot
    fig.show()
    
def plot_raw_events_xyt_w_edges_plotly(events, timespan=1500, out_path=None):
    """
    Plot a 3d scatter plot for a input raw events, with edges between events.
    Args:
        events: (N, 4) numpy array, each row is an event with format [t, x, y, p]
        timespan: int, the time span of the plot.
        out_path: str, the path to save the plot. If None, the plot will not be saved.
    """
    # Prepare data
    color = events[:,3]
    t = events[:,0]
    x = events[:,1]
    y = 260-events[:,2]
    points = np.stack([x, y, t], axis=1)

    # Parameters
    radius = 20
    max_connections = 3
    t_scale = 0.2
    N = points.shape[0]
    # Orange color with opacity set to 0.8
    # line_color = 'rgba(242, 169, 59, 0.8)'
    line_color = 'rgba(100, 100, 100, 0.8)'

    # Initialize an empty list to store edges
    edges = []

    # Calculate radius graph
    points_scaled = points.copy()
    points_scaled[:, -1] = points_scaled[:, -1] * t_scale
    for i in range(N):
        distances = np.linalg.norm(points_scaled - points_scaled[i], axis=1)
        sorted_indices = np.argsort(distances)
        num_connections = 0
        for j in sorted_indices[1:]:
            if distances[j] <= radius:
                if num_connections < max_connections:
                    edges.append((i, j))
                    num_connections += 1
                else:
                    break

    # Extract coordinates for plotting
    edge_x, edge_y, edge_z = [], [], []

    for edge in edges:
        x0, y0, z0 = points[edge[0]]
        x1, y1, z1 = points[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    scatter = go.Scatter3d(
        x=x, y=y, z=t,
        mode='markers',
        marker=dict(
            size=2,
            color=color,                # set color to an array/list of desired values
            # self define a discrete color sequence for 0 and 1 (0: lightgreen, 1: lightcoral)
            colorscale=[[0, 'green'], [1, 'coral']],
            # colorscale='Viridis',   # choose a colorscale
            opacity=1
        )
    )

    # Create 3D line plot for edges
    lines = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        # opacity of line set to 0.5
        line=dict(color=line_color, width=2.5)
    )

    camera = {'center': {'x': 0, 'y': 0, 'z': 0},
                'eye': {'x': 1.7880693412749016, 'y': -1.0138854791406924, 'z': 0.6799590178682735},
                'projection': {'type': 'orthographic'},
                'up': {'x': -0.4536565062142194, 'y': -0.8826307319128543, 'z': -0.12312093831874119}}

    layout = go.Layout(
        scene=dict(
            # label font size 16
            xaxis=dict(nticks=4, range=[0, 346], title='X', tickfont=dict(size=14), titlefont=dict(size=20)),
            yaxis=dict(nticks=4, range=[0, 260], title='Y', tickfont=dict(size=14), titlefont=dict(size=20)),
            zaxis=dict(nticks=4, range=[0, timespan], title='Time (μs)', tickfont=dict(size=14), titlefont=dict(size=20)),
        ),
        width=1000,
        height=600,
        margin=dict(r=20, l=10, b=10, t=10),
        scene_aspectratio={'x': 1, 'y': 0.75, 'z': 2},
        scene_camera=camera,
    )

    fig = go.Figure(data=[lines, scatter], layout=layout)

    # save the figure
    if out_path is not None:
        pio.write_image(fig, out_path, width=1200, height=800, scale=4)

    ## Uncomment if you want to extract the inner parameters from a plotly figure
    # f2 = go.FigureWidget(fig)
    # f2
    
    # Show plot
    fig.show()