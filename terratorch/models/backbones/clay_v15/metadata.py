"""Clay v1.5 sensor metadata: GSD (m) and per-band mean, std and wavelength (um).

From configs/metadata.yaml of https://github.com/Clay-foundation/model (Apache-2.0).
"""

CLAY_METADATA = {
    "sentinel-2-l2a": {
        "band_order": ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"],
        "rgb_indices": [2, 1, 0],
        "gsd": 10.0,
        "bands": {
            "mean": {
                "blue": 1105.0, "green": 1355.0, "red": 1552.0, "rededge1": 1887.0, "rededge2": 2422.0,
                "rededge3": 2630.0, "nir": 2743.0, "nir08": 2785.0, "swir16": 2388.0, "swir22": 1835.0,
            },
            "std": {
                "blue": 1809.0, "green": 1757.0, "red": 1888.0, "rededge1": 1870.0, "rededge2": 1732.0,
                "rededge3": 1697.0, "nir": 1742.0, "nir08": 1648.0, "swir16": 1470.0, "swir22": 1379.0,
            },
            "wavelength": {
                "blue": 0.493, "green": 0.56, "red": 0.665, "rededge1": 0.704, "rededge2": 0.74, "rededge3": 0.783,
                "nir": 0.842, "nir08": 0.865, "swir16": 1.61, "swir22": 2.19,
            },
        },
    },
    "planetscope-sr": {
        "band_order": ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"],
        "rgb_indices": [5, 3, 1],
        "gsd": 5.0,
        "bands": {
            "mean": {
                "coastal_blue": 1720.0, "blue": 1715.0, "green_i": 1913.0, "green": 2088.0, "yellow": 2274.0,
                "red": 2290.0, "rededge": 2613.0, "nir": 3970.0,
            },
            "std": {
                "coastal_blue": 747.0, "blue": 698.0, "green_i": 739.0, "green": 768.0, "yellow": 849.0, "red": 868.0,
                "rededge": 849.0, "nir": 914.0,
            },
            "wavelength": {
                "coastal_blue": 0.443, "blue": 0.49, "green_i": 0.531, "green": 0.565, "yellow": 0.61, "red": 0.665,
                "rededge": 0.705, "nir": 0.865,
            },
        },
    },
    "landsat-c2l1": {
        "band_order": ["red", "green", "blue", "nir08", "swir16", "swir22"],
        "rgb_indices": [0, 1, 2],
        "gsd": 30.0,
        "bands": {
            "mean": {
                "red": 10678.0, "green": 10563.0, "blue": 11083.0, "nir08": 14792.0, "swir16": 12276.0,
                "swir22": 10114.0,
            },
            "std": {
                "red": 6025.0, "green": 5411.0, "blue": 5468.0, "nir08": 6746.0, "swir16": 5897.0, "swir22": 4850.0,
            },
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir08": 0.86, "swir16": 1.6, "swir22": 2.2},
        },
    },
    "landsat-c2l2-sr": {
        "band_order": ["red", "green", "blue", "nir08", "swir16", "swir22"],
        "rgb_indices": [0, 1, 2],
        "gsd": 30.0,
        "bands": {
            "mean": {
                "red": 13705.0, "green": 13310.0, "blue": 12474.0, "nir08": 17801.0, "swir16": 14615.0,
                "swir22": 12701.0,
            },
            "std": {
                "red": 9578.0, "green": 9408.0, "blue": 10144.0, "nir08": 8277.0, "swir16": 5300.0, "swir22": 4522.0,
            },
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir08": 0.86, "swir16": 1.6, "swir22": 2.2},
        },
    },
    "naip": {
        "band_order": ["red", "green", "blue", "nir"],
        "rgb_indices": [0, 1, 2],
        "gsd": 1.0,
        "bands": {
            "mean": {"red": 110.16, "green": 115.41, "blue": 98.15, "nir": 139.04},
            "std": {"red": 47.23, "green": 39.82, "blue": 35.43, "nir": 49.86},
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir": 0.842},
        },
    },
    "linz": {
        "band_order": ["red", "green", "blue"],
        "rgb_indices": [0, 1, 2],
        "gsd": 0.5,
        "bands": {
            "mean": {"red": 89.96, "green": 99.46, "blue": 89.51},
            "std": {"red": 41.83, "green": 36.96, "blue": 31.45},
            "wavelength": {"red": 0.635, "green": 0.555, "blue": 0.465},
        },
    },
    "sentinel-1-rtc": {
        "band_order": ["vv", "vh"],
        "gsd": 10.0,
        "bands": {
            "mean": {"vv": -12.113, "vh": -18.673},
            "std": {"vv": 8.314, "vh": 8.017},
            "wavelength": {"vv": 3.5, "vh": 4.0},
        },
    },
    "modis": {
        "band_order": [
            "sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04", "sur_refl_b05", "sur_refl_b06",
            "sur_refl_b07",
        ],
        "rgb_indices": [0, 3, 2],
        "gsd": 500.0,
        "bands": {
            "mean": {
                "sur_refl_b01": 1072.0, "sur_refl_b02": 1624.0, "sur_refl_b03": 931.0, "sur_refl_b04": 1023.0,
                "sur_refl_b05": 1599.0, "sur_refl_b06": 1404.0, "sur_refl_b07": 1051.0,
            },
            "std": {
                "sur_refl_b01": 1643.0, "sur_refl_b02": 1878.0, "sur_refl_b03": 1449.0, "sur_refl_b04": 1538.0,
                "sur_refl_b05": 1763.0, "sur_refl_b06": 1618.0, "sur_refl_b07": 1396.0,
            },
            "wavelength": {
                "sur_refl_b01": 0.645, "sur_refl_b02": 0.858, "sur_refl_b03": 0.469, "sur_refl_b04": 0.555,
                "sur_refl_b05": 1.24, "sur_refl_b06": 1.64, "sur_refl_b07": 2.13,
            },
        },
    },
    "satellogic-MSI-L1D": {
        "band_order": ["red", "green", "blue", "nir"],
        "rgb_indices": [0, 1, 2],
        "gsd": 1.0,
        "bands": {
            "mean": {"red": 1451.54, "green": 1456.54, "blue": 1543.22, "nir": 2132.68},
            "std": {"red": 995.48, "green": 771.29, "blue": 708.86, "nir": 1236.71},
            "wavelength": {"red": 0.64, "green": 0.545, "blue": 0.48, "nir": 0.825},
        },
    },
}
