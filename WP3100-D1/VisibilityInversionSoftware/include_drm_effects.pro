;+
;
; NAME:
;   include_drm_effects.pro
;
; PURPOSE:
;     compute the drm to be included in the Bremsstrahlung equation for the inversion of count
;     visibility spectra
;
; CALLING SEQUENCE:
;       include_drm_effects,visib,srm
;
; CALLED BY:
;   - visibilities_inversion.pro
;
; INPUTS:
;   visib :      - Array of count visibility structures.
;
; OUTPUTS
;   srm:         - drm taking into account of diagonal and non-diagonal terms
;
;+

pro include_drm_effects, detOK, visib, srm

;;;;;;;; Check for the count energy ranges of the visibility bag
epsmin=min(visib.erange[0])
deltaeps=min(visib.erange[1])-epsmin
neps=(max(visib.erange[1])-epsmin)/deltaeps

;;;;;;;; Observation time
time_interval=visib[0].trange

o=hsi_calib_eventlist(DET_INDEX_MASK=detOK, $
                im_time_interval=time_interval, $
                time_range=fltarr(2), $
                im_energy=findgen(neps+1)*deltaeps+epsmin)
ptim, o->get(/obs_time)

cbd=o->getdata()

effstr=o->get(/cbe_det_eff)

;to compute the drm  for this eqtn:  C = drm # ( P * dE) - c is in cnts/cm2/keV, P in ph/cm2/keV, dE is ph energy bin width

;Get the energy bins
b = o->get(/obj, class='hsi_binned_eventlist')
ct_edges = b->getaxis(/energy, /edges_2)

;We need to use only the non-diagonal terms of sub-matrix 3.
chkarg,'hessi_build_srm

; We want to use CT_EDGES, PH_EDGES, USE_VIRD,/SEP_DETS,/SEP_VIRDS, SIMPLIFY, /PHA_ON_ROW, TIME_WANTED, ATTEN_STATE
; Return, SRM and GEOM_AREA

time_wanted = (o->get(/absolute_time))[0]
ptim, time_wanted ;only used if component 7 is used

use_vird = intarr(18)
use_vird[where(o->get(/det_index_mask))]=1

;;;;;;;;;;;;;;;;;;;; Simplify=3 means DRM=I (no effect is taken into account)
;;;;;;;;;;;;;;;;;;;; Simplify=2 means only diagonal terms are taken into account
;;;;;;;;;;;;;;;;;;;; Simplify=0 means full treatment

;;;;;;;;; Step 1: (Simplify=3) No effect is taken into account: DRM=I

simplify = intarr(10)+3
ct_edges = b->getaxis(/energy, /edges_1) ; form used by hessi_build_srm

atten_state= o->get(/image_atten_state)

hessi_build_srm, ct_edges, use_vird, srmnoeff, geom_area, /sep_dets, /sep_virds, simplify=simplify, $
    /pha_on_row, time_wanted=time_wanted, ph_edges=ct_edges, atten_state=atten_state

;;;;;;;;; Step 2: (Simplify for effect 3 = 0) Full treatment of effect 3

simplify[[3]] = 0

hessi_build_srm, ct_edges, use_vird, srm3full, geom_area, /sep_dets, /sep_virds, simplify=simplify, $
    /pha_on_row, time_wanted=time_wanted, ph_edges=ct_edges, atten_state=atten_state

;;;;;;;;; Step 3: (Simplify for effect 3 = 2) Only diagonal terms of effect 3 are taken into account

simplify[[3]] = 2

hessi_build_srm, ct_edges, use_vird, srm3diag, geom_area, /sep_dets, /sep_virds, simplify=simplify, $
    /pha_on_row, time_wanted=time_wanted, ph_edges=ct_edges, atten_state=atten_state

;;;;;;;;; STEP 4: To compute only the non-diagonal terms of effect 3

srm=srm3full-srm3diag+srmnoeff

end