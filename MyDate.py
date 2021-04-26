import argparse
import datetime
from datetime import timedelta


def check_strdate(value):
    if value is not None and len(value) == 10 and \
            value.count("-") == 2 and \
            value[:4].isnumeric() and \
            value[5:7].isnumeric() and \
            value[8:].isnumeric():
        return value
    else:
        raise argparse.ArgumentTypeError(
            "%s is an invalid format or type. Must be a string formatted as YYYY-MM-DD" % value)


class MyDate:

    def __init__(self, strdate=None, pydate=None):
        if strdate is not None and check_strdate(strdate):
            self.pydate = datetime.datetime.strptime(strdate, '%Y-%m-%d')
            self.strdate = strdate
        elif pydate is not None and isinstance(pydate, datetime.date):
            self.pydate = pydate
            self.strdate = pydate.strftime('%Y-%m-%d')
        else:
            self.pydate = None
            self.strdate = ""

    def __str__(self):
        return self.strdate

    def increment(self, add=1):
        self.pydate = self.pydate + timedelta(days=add)
        self.strdate = self.pydate.strftime('%Y-%m-%d')

    def __sub__(self, other):
        """
        Returns the difference in DAYS
        :param other:
        :return:
        """
        if isinstance(other, MyDate):
            return (self.pydate - other.pydate).days
        elif isinstance(other, datetime.date):
            return (self.pydate - other).days
        elif isinstance(other, int):
            return (self.pydate - timedelta(days=other)).days
        else:
            raise NotImplementedError(
                "MyDate : the subtraction is implemented only for MyDate-int MyDate-datetime.date, Mydate-MyDate")
