from pysolar.solar import get_altitude, get_azimuth
import datetime
import urllib.parse
import requests
from sandbox import set_logger
logger = set_logger(__name__)


class LightSource:
    """
    Get the altitude and azimuth of the sun for an specific latitude, longitude and time.
    """
    def __init__(self,
                 latitude: float = 50.779170300000004,
                 longitude: float = 6.068920799008829,
                 date: datetime.datetime = datetime.datetime(2021, 3, 19, 15, 13, 1, 130320,
                                                             tzinfo=datetime.timezone.utc)):
        """

        Args:
            latitude:
            longitude:
            date: datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
        """
        self.latitude_deg = latitude
        self.longitude_deg = longitude
        self.date = date
        self.address = 'RWTH Aachen, Germany'
        logger.info("LightSource set to address %s at datetime %s" % (self.address, self.date.ctime()))


    def update(self, sb_params: dict):
        sb_params["altitude"] = self.altitude
        sb_params["azimuth"] = self.azimuth
        return sb_params

    @property
    def url(self):
        """
        Search addres in web page: https://nominatim.openstreetmap.org/ui/search.html
        Returns:
            url address
        """
        return 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(self.address) + '?format=json'

    @property
    def altitude(self):
        """ Get the altitude of the sun position based on the latitude, longitude and date"""
        return get_altitude(self.latitude_deg, self.longitude_deg, self.date)

    @property
    def azimuth(self):
        """ Get the azimuth of the sun position based on the latitude, longitude and date"""
        return get_azimuth(self.latitude_deg, self.longitude_deg, self.date)

    def set_address(self, address: str = 'RWTH Aachen, Germany'):
        """
        Provide an address, city, or country to search for the latitude and longitude
        Args:
            address: e.g. 'Aachen', 'Germany', 'India', 'Colombia', 'RWTH Aachen University'
        Returns:

        """
        self.address = address

    def set_latitude_longitude(self):
        """
        Set the latitude and longitude based on the self.address
        Returns:
            set the self.latitude_deg and self.longitude_deg
        """
        try:
            response = requests.get(self.url).json()
            self.latitude_deg = response[0]["lat"]
            self.longitude_deg = response[0]["lon"]
        except Exception:
            logger.warning("Address '%s' not found, change address" % self.address)

