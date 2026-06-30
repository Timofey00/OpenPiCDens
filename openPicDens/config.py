"""
Config.py
"""

import os


saveExt = "txt"
saveRWLExt = "rwl"
sectorPorosityNumber = 10

SAVE_PATHS = {
	"sd_porosity_path": "areaPorosity",
	"natural_path": "naturalValuesPorosity",
	"norm_path": "normValuesPorosity",
	"raw_path": "rawPorosity",
	"sec_path": "sectorsPorosity",
	"rwl": "rwl"
}

SAVE_NAMES = {
	"rw": f"rw.{saveExt}",
	"max": f"maxPorosity.{saveExt}",
	"mean": f"meanPorosity.{saveExt}",
	"min": f"minPorosity.{saveExt}",
	"minQ": f"minPorosityQ.{saveExt}",
	"maxQ": f"maxPorosityQ.{saveExt}",
	"meanQ": f"meaPorosityQ.{saveExt}",
	"ew": f"eww.{saveExt}",
	"lw": f"lww.{saveExt}",
	"lwpr": f"lwPerc.{saveExt}",
	"ewpr": f"ewPerc.{saveExt}",
	"ewp": f"ewPorosity.{saveExt}",
	"lwp": f"lwPorosity.{saveExt}",
	"long": f"long.{saveExt}",
	"zlong": f"zlong.{saveExt}",
	"avg": f"avg.{saveExt}",
}

SAVE_RWL_NAMES = {
	"rw": f"rw.{saveRWLExt}",
	"max": f"maxPorosity.{saveRWLExt}",
	"mean": f"meanPorosity.{saveRWLExt}",
	"min": f"minPorosity.{saveRWLExt}",
	"minQ": f"minPorosityQ.{saveRWLExt}",
	"maxQ": f"maxPorosityQ.{saveRWLExt}",
	"meanQ": f"meaPorosityQ.{saveRWLExt}",
	"ew": f"eww.{saveRWLExt}",
	"lw": f"lww.{saveRWLExt}",
	"lwpr": f"lwPerc.{saveRWLExt}",
	"ewpr": f"ewPerc.{saveRWLExt}",
	"ewp": f"ewPorosity.{saveRWLExt}",
	"lwp": f"lwPorosity.{saveRWLExt}",
}

SAVE_NAMES.update({sec: f"sector_{sec}.{saveExt}" for sec in range(sectorPorosityNumber)})

SAVE_SETTINGS = {
	"txt_files_separator": '\t'
}