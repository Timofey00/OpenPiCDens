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
	"lw": f"eww.{saveExt}",
	"ew": f"lww.{saveExt}",
	"lwpr": f"lwPerc.{saveExt}",
	"ewpr": f"ewPerc.{saveExt}",
	"ewp": f"ewPorosity.{saveExt}",
	"lwp": f"lwPorosity.{saveExt}",
	"long": f"long.{saveExt}",
	"avg": f"avg.{saveExt}",
}

SAVE_RWL_NAMES = {
	"rw": f"rw.{saveRWLExt}",
	"max": f"maxPorosity.{saveExt}",
	"mean": f"meanPorosity.{saveExt}",
	"min": f"minPorosity.{saveExt}",
	"minQ": f"minPorosityQ.{saveExt}",
	"maxQ": f"maxPorosityQ.{saveExt}",
	"meanQ": f"meaPorosityQ.{saveExt}",
	"lw": f"eww.{saveExt}",
	"ew": f"lww.{saveExt}",
	"lwpr": f"lwPerc.{saveExt}",
	"ewpr": f"ewPerc.{saveExt}",
	"ewp": f"ewPorosity.{saveExt}",
	"lwp": f"lwPorosity.{saveExt}",
}

SAVE_NAMES.update({sec: f"sector_{sec}.{saveExt}" for sec in range(sectorPorosityNumber)})

SAVE_SETTINGS = {
	"txt_files_separator": '\t'
}