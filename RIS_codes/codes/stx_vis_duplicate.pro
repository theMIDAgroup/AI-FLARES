;+
; Procedure:
;   Given an array of combined STIX visibilities returns a new array with duplication of uv points
;   (mirrored with respect to the origin of the (u,v)-plane). The mirrored visibility values are computed
;   by converting the corresponding visibilities into their conjugate values.
;
; Usage:
;   visout = stx_vis_duplicate(visin)
;
; Input:
; visin is an array of STIX visibility structures.
;
; Output:
; visout is an array of STIX visibility structures including those mirrored with respect to the origin of the
; (u,v)-plane.
;
; Modification History:
;    July 2023, Perracchione E., first released (inspired from hsi_vis_duplicate)

FUNCTION stx_vis_duplicate, visin

  ;;;;;;;;;;;;;;;;; Check if duplication is necessary ;;;;;;;;;;;;;;;;;

  cntr_vis = N_elements(visin)

  ;;;;;;;;;;;;;;;;; If duplication is not necessary just return the input visibilities ;;;;;;;;;;;;;;;;;
  isc = visin.isc
  uniq_isc = isc[UNIQ(isc, SORT(isc))]
  if (n_elements(uniq_isc) EQ cntr_vis/2.) then RETURN, visin

  ;;;;;;;;;;;;;;;;; If duplication is necessary, create a new array of visibility structures ;;;;;;;;;;;;;;;;;
  
  visout = replicate(stx_visibility(),2*cntr_vis)

  ;;;;;;;;;;;;;;;;; In the first half positions copy the input visibilities ;;;;;;;;;;;;;;;;;
  
  visout[0:cntr_vis-1] = visin

  ;;;;;;;;;;;;;;;;; In the second half positions duplicate the visibilities ;;;;;;;;;;;;;;;;;

  visout[cntr_vis:2*cntr_vis-1].u              = -visin.u
  visout[cntr_vis:2*cntr_vis-1].v              = -visin.v
  visout[cntr_vis:2*cntr_vis-1].obsvis         = conj(visin.obsvis)
  visout[cntr_vis:2*cntr_vis-1].isc            = visin.isc
  visout[cntr_vis:2*cntr_vis-1].label          = visin.label
  visout[cntr_vis:2*cntr_vis-1].ENERGY_RANGE   = visin.ENERGY_RANGE
  visout[cntr_vis:2*cntr_vis-1].time_range     = visin.time_range
  visout[cntr_vis:2*cntr_vis-1].totflux        = visin.totflux
  visout[cntr_vis:2*cntr_vis-1].sigamp         = visin.sigamp
  visout[cntr_vis:2*cntr_vis-1].xyoffset       = visin.xyoffset
  visout[cntr_vis:2*cntr_vis-1].calibrated     = visin.calibrated
  visout[cntr_vis:2*cntr_vis-1].phase_sense    = visin.phase_sense
  visout[cntr_vis:2*cntr_vis-1].live_time      = visin.live_time
  
  return, visout
    
end