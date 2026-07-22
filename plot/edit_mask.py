import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def main(nc_path):

    nc = Dataset(nc_path, 'r+')

    if 'mask_rho' not in nc.variables:
        print("Error: 'mask_rho' not found.")
        return

    mask_var = nc.variables['mask_rho']
    mask = mask_var[:]

    fig, ax = plt.subplots()
    im = ax.imshow(mask, origin='lower', cmap='Blues', interpolation='nearest')

    ax.set_title(
        "Zoom normally with toolbar.\n"
        "Hold SHIFT + Left Click to toggle cell.\n"
        "Press 's' to save, 'q' to quit."
    )

    plt.colorbar(im, ax=ax, label="mask_rho")

    changed = False

    def onclick(event):
        nonlocal changed

        # Only respond if inside axes
        if event.inaxes != ax:
            return

        # Only toggle if SHIFT is pressed
        if event.key != 'shift':
            return

        # If toolbar mode is active (zoom or pan), do nothing
        if fig.canvas.manager.toolbar.mode != '':
            return

        if event.xdata is None or event.ydata is None:
            return

        i = int(np.floor(event.xdata))
        j = int(np.floor(event.ydata))

        if 0 <= j < mask.shape[0] and 0 <= i < mask.shape[1]:
            mask[j, i] = 1 - mask[j, i]
            im.set_data(mask)
            fig.canvas.draw_idle()
            changed = True
            print(f"Toggled cell ({j}, {i}) → {mask[j,i]}")

    def onkey(event):
        nonlocal changed

        if event.key == 's':
            if changed:
                mask_var[:] = mask
                nc.sync()
                print("Changes saved.")
                changed = False
            else:
                print("No changes to save.")

        elif event.key == 'q':
            print("Closing.")
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()
    nc.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python edit_mask_rho_visual_zoom.py <roms_grid_file.nc>")
    else:
        main(sys.argv[1])
