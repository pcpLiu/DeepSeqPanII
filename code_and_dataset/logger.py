import logging


def setup_logging(output_folder, out_file='log.txt'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(name)s]: %(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename='{}/{}'.format(output_folder, out_file),
        filemode='w'
    )

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(name)s]: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def log_to_file(name, value):
    logger = logging.getLogger(name)
    logger.info(value)


if __name__ == '__main__':
    pass