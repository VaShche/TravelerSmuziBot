import datetime


def getNow(l=-7):
    """
    :param l: integer
    l=10 returns YYYY-MM-DD;
    l=19 or l=-7 returns YYYY-MM-DD HH:MM:SS

    :return:
    """
    return str(datetime.datetime.now())[:l]


def log(*args):
    print('{}\t{}'.format(getNow(), args))
