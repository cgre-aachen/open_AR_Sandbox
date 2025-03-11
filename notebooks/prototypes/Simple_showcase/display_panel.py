#%%
import panel as pn
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.transform import linear_cmap
from bokeh.io import output_notebook
import os
#%%

top = r"""
<style>
/* Define the custom font */
@font-face {
  font-family: HelveticaNeueLT Com 47 LtCn;
  src: url('.\font\helvetica-neue-lt-com-47-light-condensed.ttf') format('truetype');
}

/* Apply the custom font to specific elements */
body {
  font-family: 'HelveticaNeueLT Com 47 LtCn', sans-serif;
}

h1 {
  font-family: 'HelveticaNeueLT Com 47 LtCn', sans-serif;
}
</style>

# Open-AR-Sandbox

---

Herzlich Willkommen an unserer besonderen Sandbox. Sie ist ein Instrument, das wir verwenden, um Studierenden der Geowissenschaften wichtige Konzepte wie räumliche Vorstellung und Kartenkunde zu vermitteln. 
<br></br>
Aber was macht diese Sandbox so besonders? Der Sand ist ganz normaler Sand, wie man ihn auf dem Spielplatz findet. Oben befinden sich ein Beamer wie im Heimkino. Außerdem findet man oben einen "Kinect" Sensor (Microsoft). Dieser sendet infrarot Lichtsignale auf die Sandoberfläche und nimmt diese wieder auf. Diese Methode heißt "Time-of-flight", also wird die Zeit zwischen Aussenden und Empfang gemessen und so eine "Tiefenkarte" erstellt.
<br></br>
Mit dieser Tiefenkarte machen wir mit unserer eigenen Software Berechnungen, die dann wieder auf den Sand projezieren. In dieser Austellung werden sie diese Themen erkunden können.
- Eine interaktive Landkarte
- Künstliche Satellitenfotos
<div></div>
Die Satellitenfotos werden von einem sogenannten <b>U-Net</b> generiert. Dieses Neurale Netzwerk wird trainiert, in dem es viele Ausschnitte von Höhenmodellen und das dazugehörige Foto von einem Satelliten zu sehen bekommt. Durch Umformuen dieser Bilder und die große Anzahl von Bildern lernt dieses Netzwerk, wie ein Satellitenbild aussehen sollte, wenn das Höhenmodell bekannt ist. Das hier ausgestellte U-Net wurde mit Bildern aus den Alpen trainiert und interpretiert die Höhe des Sandes als Höhenmodell.

"""


footer = r"""

---

### Footer

![Footer Image](.\images\logo.svg)


"""

pn.extension()
tmp_imgs = os.listdir("temp_img")


def load_dem(dem):
    return np.load("temp_img/" + dem)

def load_sat(sat):
    return np.load("temp_img/" + sat) * 255

def update_img():
    global dem
    dem_array = load_dem(dem)
    img1.data_source.data = {'image': [dem_array]}
    color_mapper.low = dem_array.min()
    color_mapper.high = dem_array.max()


    sat_im = load_sat(sat)
    xdim, ydim, _  = sat_im.shape
    sat_im_rgb = np.empty((ydim, xdim), dtype=np.uint32)
    view = sat_im_rgb.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    view[:,:,:3] = sat_im #np.flipud(sat_im)
    view[:,:,3] = 255

    img2.data_source.data = {'image': [sat_im_rgb]}

for ifile in tmp_imgs:
    if "dem" in ifile:
        dem = ifile
    if "sat_gen" in ifile:
        sat = ifile


dem_array = load_dem(dem)
sat_array = load_sat(sat)

color_mapper = LinearColorMapper(palette="Viridis256", low=dem_array.min(), high=dem_array.max())


fig1 = figure(title="Höhenmodell", plot_height=300, plot_width=300)
img1 = fig1.image(image=[load_dem(dem)], x=0, y=0, dw=10, dh=10, color_mapper=color_mapper)

fig2 = figure(title="Satellitenbild aus U-Net", plot_height=300, plot_width=300)
sat_im = load_sat(sat)
xdim, ydim, _  = sat_im.shape
sat_im_rgb = np.empty((ydim, xdim), dtype=np.uint32)
view = sat_im_rgb.view(dtype=np.uint8).reshape((ydim, xdim, 4))
view[:,:,:3] = np.flipud(sat_im)
view[:,:,3] = 255
#sat_im_rgba = sat_im_rgba.view(dtype=np.uint32).reshape((256,256))
img2 = fig2.image_rgba(image=[sat_im_rgb], x=0, y=0, dw=10, dh=10)


fig1.toolbar.logo = None
fig1.toolbar_location = None
fig2.toolbar.logo = None
fig2.toolbar_location = None

#%%
dashboard = pn.Column(
    pn.pane.Markdown(top, style={'background': '#f0f0f0', 'padding': '10px'}),
    pn.Row(fig1, fig2),
    pn.pane.Markdown(footer, style={'background': '#f0f0f0', 'padding': '10px'})

)

pn.state.add_periodic_callback(update_img, period=500)

dashboard.servable()

