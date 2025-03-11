<style>
/* Define the custom font */
@font-face {
  font-family: HelveticaNeueLT Com 47 LtCn;
  src: url('C:\Users\chudalla\Desktop\Python_repos\open_AR_Sandbox\notebooks\prototypes\Simple_showcase\font\helvetica-neue-lt-com-47-light-condensed.ttf') format('truetype');
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

![Main Image](C:\Users\chudalla\Desktop\Python_repos\open_AR_Sandbox\notebooks\prototypes\Simple_showcase\images\geomechanic.png)

---

### Footer

![Footer Image](C:\Users\chudalla\Desktop\Python_repos\open_AR_Sandbox\notebooks\prototypes\Simple_showcase\images\logo.svg)

