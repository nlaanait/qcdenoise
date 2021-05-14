import logging
__all__ = ["_plots", "nx_plot_options"]

# module logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "dataset- %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# set plotting flag
global _plots, nx_plot_options
try:
    import matplotlib
    _plots = True
    nx_plot_options = {
        'with_labels': True,
        'node_color': 'red',
        'node_size': 175,
        'width': 2,
        'font_weight': 'bold',
        'font_color': 'white',
    }
except ImportError:
    logger.warning(
        "matplotlib could not be imported- skipping plotting.")
    _plots = False
    nx_plot_options = None
