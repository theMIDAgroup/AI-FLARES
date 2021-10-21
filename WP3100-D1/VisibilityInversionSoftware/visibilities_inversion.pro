;+
;
; NAME:
;   visibilities_inversion.pro
;
; PURPOSE:
;     this is the main routine of the Visibility Inversion Software (VIS) implementing the
;     method described in Piana et al. The Astrophysical Journal, 665:846-855, 2007 to compute
;     electron visibilities from RHESSI count visibilities.
;     In particular this routine:
;     1) build count visibility spectra, for each (u,v) pair, as a function of the count energy
;     2) invert the count visibility spectra to provide regularized electron visibility spectra
;     3) save the inverted spectra in arrays of visibility structures in the same format of the
;        standard arrays of count visibility structures.
;
; CALLING SEQUENCE:
;       visibilities_inversion, vis, $
;                               reg_el_vis, orig_ph_vis, reg_ph_vis
;
; CALLED BY:
; 
;   - The user must call this routine inside his own code or in the IDL prompt.
;
; CALLS TO:
;
;   - build_count_vis_spectra.pro
;   - include_drm_effects.pro
;   - vis_regularized_inversion.pro
;   - el_spectra_2_el_vis.pro
;
; INPUTS:
;
;   vis:         array of RHESSI count visibility structures (counts cm^-2 s^-1). 
;                Each element of the array must be a RHESSI visibility structure corresponding to a specific energy range
;
; OUTPUTS
;
;   reg_el_vis:  array of electron visibility structures for the basic regularized solution (electrons cm^-2 s^-1 keV^-1)
;   orig_ph_vis: array of the sub-set of the count visibility structures used as input for the inversion (counts cm^-2 s^-1 keV^-1)
;   reg_ph_vis:  array of regularized count visibility structures corresponding to the recovered electron visbilities (counts cm^-2 s^-1 keV^-1)
;
;+

pro visibilities_inversion, vis, $
                            reg_el_vis, orig_ph_vis, reg_ph_vis

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;; Edit visibilities (if necessary)

visedit=hsi_vis_edit(vis)

;;;;;;;; Check for the detectors involved

detOK=intarr(9)
FOR isc = 0, 8 DO BEGIN
    isthere=where(visedit.isc eq isc,count)
    if (count GT 0) then detOK[isc]=1
END


;;;;;;;; Build count visibility spectra
build_count_vis_spectra, detOK, visedit, count_vis_spectra

;;;;;;;; DRM computation for the involved detectors
include_drm_effects, detOK, visedit, srm

;;;;;;;; Inversion
Nspectra=N_elements(count_vis_spectra.det)
epsmin=min(visedit.erange[0])
deltaeps=min(visedit.erange[1])-epsmin

spectraptr=ptrarr(Nspectra)

confidencestrip=10
for l=0,Nspectra-1 do begin
   if (count_vis_spectra.accepted[l] eq 1) then begin
    print, 'Inversion of spectra n.', l
    ;;;; spectrum 'l' comes from detector 'count_vis_spectra.det(l)'
    drmini=srm[*,*,total(detOK,/preserve_type)-total(detOK[count_vis_spectra.det[l]:8],/preserve_type)]

    ;;;;;;;; Real part inversion
    part='Real'
    vis_regularized_inversion, l, part, opt, count_vis_spectra, outdataRe, drmini, electron_bin=deltaeps, confidencestrip=confidencestrip

    nphRe=N_elements(outdataRe.residuals[*,0])
    neeRe=N_elements(outdataRe.strip[*,0])

    ;;;;;;;; Imaginary part inversion
    part='Imaginary'
    vis_regularized_inversion, l, part, opt, count_vis_spectra, outdataIm, drmini, electron_bin=deltaeps, confidencestrip=confidencestrip


    nphIm=N_elements(outdataIm.residuals[*,0])
    neeIm=N_elements(outdataIm.strip[*,0])

    ;;;;;;;; Save inverted spectra

    outdata={stripRe:dblarr(neeRe,confidencestrip+2),residualsRe:dblarr(nphRe,6),stripIm:dblarr(neeIm,confidencestrip+2),residualsIm:dblarr(nphIm,6)}
    outdata.stripRe=outdataRe.strip
    outdata.stripIm=outdataIm.strip
    outdata.residualsRe=outdataRe.residuals
    outdata.residualsIm=outdataIm.residuals

    spectraptr[l]=ptr_new(outdata)
    if (nphRe LE 3 or nphIm LE 3) then count_vis_spectra.accepted[l]=0

   endif
end

print, ''
for l=0,Nspectra-1 do begin
    if (count_vis_spectra.accepted[l] eq 0) then print, 'Rejected spectra n.', l
endfor

vis_el_spectra_2_el_vis, visedit, count_vis_spectra, spectraptr, reg_el_vis, orig_ph_vis, reg_ph_vis

end
