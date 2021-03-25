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
                 date: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)):
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
        self.manual = False
        self.simulation = False
        self._altitude = 45
        self._azimuth = 315
        self._ve = 0.25
        self._add_time = 1  # add 1 hour when simulation
        self.full_address = 'RWTH Aachen, Germany'
        logger.info("LightSource set to address %s at datetime %s" % (self.address, self.date.ctime()))

    @property
    def url(self):
        """
        Search address in web page: https://nominatim.openstreetmap.org/ui/search.html
        Returns:
            url address
        """
        return 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(self.address) + '?format=json'

    @property
    def altitude(self):
        """ Get the altitude of the sun position based on the latitude, longitude and date"""
        if self.manual:
            return self._altitude
        if self.simulation:
            self.add_time()
        return get_altitude(self.latitude_deg, self.longitude_deg, self.date)

    @property
    def azimuth(self):
        """ Get the azimuth of the sun position based on the latitude, longitude and date"""
        if self.manual:
            return self._azimuth
        if self.simulation:
            self.add_time()
        return get_azimuth(self.latitude_deg, self.longitude_deg, self.date)

    def add_time(self):
        self.date = self.date + datetime.timedelta(hours=self._add_time)

    @property
    def ve(self):
        """
        Vertical exaggeration
        Returns:
        """
        return self._ve

    def set_ve(self, ve):
        """
        Set vertical exaggeration
        Args:
            ve: Vertical exaggeration

        Returns:

        """
        self._ve = ve

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
            self.latitude_deg = float(response[0]["lat"])
            self.longitude_deg = float(response[0]["lon"])
            self.full_address = response[0]["display_name"]
        except:
            logger.warning("Address '%s' not found, change address" % self.address)

    def set_datetime(self, year=2021, month=3, day=19, hour=15, minute=13, second=1, date=None, time=None,
                     tzinfo=datetime.timezone.utc):
        """
        Set the datetime to acquire the sun position
        Args:
            year: int
            month: int [1-12]
            day: int [1-31]
            hour: int [0-23]
            minute: int [0-59]
            second: int [0-59]
            date: datetime.date
            time: datetime.time
            tzinfo: datetime.timezone

        Returns:

        """
        if date is not None or time is not None:
            self.date = datetime.datetime.combine(date if date is not None else self.date.date(),
                                                  time if time is not None else self.date.time(),
                                                  tzinfo=tzinfo)
        else:
            self.date = datetime.datetime(year, month, day, hour, minute, second, tzinfo=tzinfo)

    def set_altitude(self, altitude):
        """
        Set the altitude manually. self.manual must be True
        Args:
            altitude:

        Returns:

        """
        self._altitude = altitude

    def set_azimuth(self, azimuth):
        """
        Set the azimuth manually. self.manual must be True
        Args:
            azimuth:

        Returns:
        """
        self._azimuth = azimuth