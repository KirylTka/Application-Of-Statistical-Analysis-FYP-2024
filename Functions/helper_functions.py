import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def helper_cmaps(imgs): #Configures plt.imshow() cmaps for images.
    '''
    Configures plt.imshow() cmap settings for plotting. Matches the vmax and vmin across all images
    Use like this: plt.imshow(img, **helper_cmaps(imgs))
    Or if single image: plt.imshow(img, **helper_cmaps([img]))
    imgs: list of images
    
    '''
    imgs = np.concatenate([np.array(img).flatten() for img in imgs])
    min_val = np.nanmin(imgs)
    max_val = np.nanmax(imgs)
    # pos = dict(cmap=mpl.colormaps['Greys'],vmin=0,vmax=np.nanmax(imgs))
    pos = dict(cmap=mpl.colormaps['viridis'],vmin=0,vmax=np.nanmax(imgs))
    pos_and_neg = dict(cmap=mpl.colormaps['bwr'],vmin=min_val,vmax=max_val)
    if min_val<0:
        return pos_and_neg
    else:
        return pos
    
def pims(imgs,title=None,figsize = (10,6)):
    '''
    Plotting Helper Function for plotting multiple figures at once
    Plots up to 15 images from a list
    imgs: list of images
    title: title of plot
    figsize: figsize of plot
    '''
    fig, axs = plt.subplots(nrows=3,ncols=5,layout='constrained',figsize=figsize)
    fig.patch.set_facecolor((211/255,238/255,251/255,1))
    for ax in axs.ravel():
        ax.set_axis_off()
    imgs = imgs[0:15]
    for i,img in enumerate(imgs):
        # axs.ravel()[i].set_axis_on()
        im = axs.ravel()[i].imshow(img,**helper_cmaps(imgs))
        axs.ravel()[i].set_title(str(i+1))
        fig.suptitle(title)
    fig.colorbar(im, ax = axs.ravel().tolist(),shrink=0.2,orientation='horizontal')
    plt.show(block=False)

def to_dist(imgs):
    '''
    Converts a list of images to a distribution
    imgs: list of images
    '''
    oup = []
    for im in imgs:
        image = im.copy()
        image = image[image!=0]
        image = image[~np.isnan(image)]
        image = image.flatten()
        oup.append(image)
    return np.concatenate(oup)

def plot_dists(dist_h,dist_uh,bin_n,labels=['Healthy','HCM'],ax = None):
    '''
    Plot Distributions obtained from to_dist(imgs)
    dist_h,dist_uh: the two distributions you want to plot. (N,) Numpy array of pixel values
    bin_n: number of bins to use in histogram
    labels: labels assigned to dist_h,dist_uh respectively
    ax: axis to plot to, if no axis set, new axis created
    '''
    if ax is None:
        fig,ax = plt.subplots()
    # fig.patch.set_facecolor((211/255,238/255,251/255,1))
    bins = np.linspace(min(dist_h.min(),dist_uh.min()),max(dist_h.max(),dist_uh.max()),bin_n,endpoint=True)
    dist_hy, h_bin_edges = np.histogram(dist_h,bins,density=True)
    dist_uhy, uh_bin_edges = np.histogram(dist_uh,bins,density=True)
    bincenters = 0.5 * (h_bin_edges[1:] + h_bin_edges[:-1])
    ax.hist(dist_h,bins=bins,color='g',alpha=0.3,density=True,label=labels[0])
    ax.hist(dist_uh,bins=bins,color='r',alpha=0.3,density=True,label=labels[1])
    ax.plot(bincenters,dist_hy,'-g',lw=1)
    ax.plot(bincenters,dist_uhy,'-r',lw=1)
    # ax.set_xlabel('Local Variance')
    # ax.set_ylabel('Density Frequency')
    ax.legend()

def plot_ims(set_nan=True, *args):
    '''
    Another Plotting Function Helper
    set_nan: sets the 0 values in the image to nan so no background is present.
    args: list of images.
    '''
    N = len(args)
    if N < 3:
        ncols = N
    else:
        ncols = 3
    fig,axs = plt.subplots(nrows = (N-1)//3+1,ncols = ncols)
    # fig.patch.set_facecolor((211/255,238/255,251/255,1))
    # fig.patch.set_facecolor((0,0,0,1))
    if N > 1:
        for ax in axs.ravel():
            ax.set_axis_off()
        for ax,img in zip(axs.ravel(),args):
            ax.set_axis_off()
            img = img.copy()
            img = img.astype(float)
            if set_nan: img[img==0] = np.nan

            # cmap = helper_cmaps([img])['cmap']
            # colors = [cmap(i) for i in np.linspace(0,1,int(np.nanmax(img)+1))]
            # cmaps = mcolors.ListedColormap(colors)
            # bounds = [i for i in range(int(np.nanmax(img)+2))]
            # norm = mcolors.BoundaryNorm(bounds, cmaps.N)
            # im = ax.imshow(img,cmap=cmaps,norm=norm)
            im = ax.imshow(img,**helper_cmaps(args))
            # im = ax.imshow(img,cmap=helper_cmaps(args)['cmap'],vmin=-90,vmax=90)
        cbar = fig.colorbar(im, ax = axs.ravel().tolist(),shrink=0.3,orientation='horizontal',ticks = [0,180,360])
        # cbar = fig.colorbar(im, ax = axs.ravel().tolist(),shrink=0.5,orientation='horizontal',ticks=[-90,-45,0,45,90],label='TA')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Circumfrential Position',size=16)
    else:
        img = args[0]
        axs.set_axis_off()
        img = img.copy()
        img = img.astype(float)
        if set_nan: img[img==0] = np.nan
        # im = axs.imshow(img,cmap=helper_cmaps(args)['cmap'],vmin=-90,vmax=90)
        # cmap = helper_cmaps(args)['cmap']
        # colors = [cmap(i) for i in np.linspace(0,1,int(np.nanmax(img)+1))]
        # cmaps = mcolors.ListedColormap(colors)
        # bounds = [i for i in range(int(np.nanmax(img)+2))]
        # norm = mcolors.BoundaryNorm(bounds, cmaps.N)

        # im = axs.imshow(img,**helper_cmaps([img]))
        im = axs.imshow(img,helper_cmaps([img])['cmap'],vmin=-90,vmax=90)
        # im = axs.imshow(img,cmap=cmaps,norm=norm)
        # cbar = fig.colorbar(im, ax = axs,shrink=0.3,orientation='horizontal',spacing='proportional',ticks=np.linspace(0,int(np.nanmax(img)),int(np.nanmax(img)+1)))
        # cbar = fig.colorbar(im, ax = axs,shrink=0.5,orientation='horizontal',ticks=[0,180,360])
        cbar = fig.colorbar(im, ax = axs,shrink=0.5,orientation='horizontal',ticks=[-90,-45,0,45,90])
        # cbar = fig.colorbar(im, ax = axs,shrink=0.5,orientation='horizontal')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('E2A (degrees)',size=16)
    plt.show()

    