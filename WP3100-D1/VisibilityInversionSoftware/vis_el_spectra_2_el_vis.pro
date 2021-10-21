;+
;
; NAME:
;   el_spectra_2_el_vis.pro
;
; PURPOSE:
;     take in input the regularized visibility electron spectra and rearrange
;     the electron visibilities in standard array of visibility structures.
;
; CALLING SEQUENCE:
;     el_spectra_2_el_vis, visib, countspectra, spectraptr, reg_el_vis, count_vis, reg_count_vis
;
; CALLED BY:
;
;   - visibilities_inversion.pro
;
; CALLS TO:
;
;   - vis_converter.pro
;
; INPUTS:
;
;   visib:           - Array of count visibility structures.
;   countspectra:    - A 9-tag structure containing all the information about the count visibility spectra inverted by the software
;   spectraptr       - A pointer to 2-tag structures containing all information about the inverted regularized electron spectra
;
; OUTPUTS
;
;   reg_el_vis:  array of electron visibility structures (for the basic regularized solution)
;   stripvisptr: array of pointers to arrays of electron visibility structures (for each solution in the strip of confidence)
;   orig_ph_vis: array of the count visibility structures used as input for the inversion
;   reg_ph_vis:  array of regularized count visibility structures corresponding to the recovered electron visbilities
;
;+

pro vis_el_spectra_2_el_vis, visib, countspectra, spectraptr, reg_el_vis, count_vis, reg_count_vis


;;;;;;;; Check for the count energy ranges of the visibility bag
epsmin=min(visib.erange[0])
deltaeps=min(visib.erange[1])-epsmin
neps=(max(visib.erange[1])-epsmin)/deltaeps

;;;;;;;; Count energy sampling
eps=epsmin+deltaeps*findgen(neps)

;;;;;;;;
ee=eps[0]+deltaeps/2.
Nspectra=N_elements(countspectra.det)

viscont=1

while (viscont ne 0) do begin
viscont=0
for i=0,Nspectra-1 do begin
 if (countspectra.accepted[i] eq 1) then begin
    outdata=*spectraptr[i]
    ind_e_re=where(outdata.stripRe[*,0] eq ee,countRe)
    ind_e_im=where(outdata.stripIm[*,0] eq ee,countIm)
    if (countRe eq 1 and countIm eq 1) then viscont=viscont+1
    if (viscont eq 1 and countRe eq 1 and countIm eq 1) then begin
        u=countspectra.uv[0,i]
        v=countspectra.uv[1,i]
        realvis=outdata.stripRe[ind_e_re,1]
        imvis=outdata.stripIm[ind_e_im,1]
        errRe=0.5*(max(outdata.stripRe[ind_e_re,2:11])-min(outdata.stripRe[ind_e_re,2:11]))
        errIm=0.5*(max(outdata.stripIm[ind_e_im,2:11])-min(outdata.stripIm[ind_e_im,2:11]))
        err=1./sqrt(2)*sqrt(errRe^2. + errIm^2.)
    endif
    if (viscont GT 1 and countRe eq 1 and countIm eq 1) then begin
        u=[u,countspectra.uv[0,i]]
        v=[v,countspectra.uv[1,i]]
        realvis=[realvis, outdata.stripRe[ind_e_re,1]]
        imvis=[imvis, outdata.stripIm[ind_e_im,1]]
        errRe=0.5*(max(outdata.stripRe[ind_e_re,2:11])-min(outdata.stripRe[ind_e_re,2:11]))
        errIm=0.5*(max(outdata.stripIm[ind_e_im,2:11])-min(outdata.stripIm[ind_e_im,2:11]))
        err=[err, 1./sqrt(2)*sqrt(errRe^2. + errIm^2.)]
    endif
 endif
end
if (ee eq eps[0]+deltaeps/2.) then begin
    reg_el_vis=vis_converter(ee-deltaeps/2.,ee+deltaeps/2., u, v, realvis, imvis, err, visib[0].trange, visib[0].xyoffset)
endif else begin
    visE=vis_converter(ee-deltaeps/2.,ee+deltaeps/2., u, v,  realvis,  imvis,  err, visib[0].trange, visib[0].xyoffset)
    reg_el_vis=[reg_el_vis,visE]
endelse
ee=ee+deltaeps
endwhile

;;;;;;;;;;; Repeat for count visibilities

for ieps=0,neps-1 do begin
    viscont=0
    for i=0,Nspectra-1 do begin
     if (countspectra.accepted[i] eq 1) then begin
        outdata=*spectraptr[i]
        ind_eps_re=where(outdata.residualsRe[*,0] eq eps[ieps]+deltaeps/2.,countRe)
        ind_eps_im=where(outdata.residualsIm[*,0] eq eps[ieps]+deltaeps/2.,countIm)
        if (countRe eq 1 and countIm eq 1) then viscont=viscont+1
        if (viscont eq 1 and countRe eq 1 and countIm eq 1) then begin
            u=countspectra.uv[0,i]
            v=countspectra.uv[1,i]
            countrealvis=outdata.residualsRe[ind_eps_re,1]
            countimvis=outdata.residualsIm[ind_eps_im,1]
            err=outdata.residualsRe[ind_eps_re,2]
            regcountRe=outdata.residualsRe[ind_eps_re,3]
            regcountIm=outdata.residualsIm[ind_eps_im,3]
        endif
        if (viscont GT 1 and countRe eq 1 and countIm eq 1) then begin
            u=[u,countspectra.uv[0,i]]
            v=[v,countspectra.uv[1,i]]
            countrealvis=[countrealvis, outdata.residualsRe[ind_eps_re,1]]
            countimvis=[countimvis, outdata.residualsIm[ind_eps_im,1]]
            err=[err, outdata.residualsRe[ind_eps_re,2]]
            regcountRe=[regcountRe, outdata.residualsRe[ind_eps_re,3]]
            regcountIm=[regcountIm, outdata.residualsIm[ind_eps_im,3]]
        endif
      endif
     end
     if (ieps eq 0) then begin
        count_vis=vis_converter(eps[ieps],eps[ieps]+deltaeps, u, v,  countrealvis,  countimvis,  err, visib[0].trange, visib[0].xyoffset)
        reg_count_vis=vis_converter(eps[ieps],eps[ieps]+deltaeps, u, v,  regcountRe,  regcountIm,  err, visib[0].trange, visib[0].xyoffset)
     endif else begin
        viscount=vis_converter(eps[ieps],eps[ieps]+deltaeps, u, v,  countrealvis,  countimvis,  err, visib[0].trange, visib[0].xyoffset)
        reg_viscount=vis_converter(eps[ieps],eps[ieps]+deltaeps, u, v,  regcountRe,  regcountIm,  err, visib[0].trange, visib[0].xyoffset)
        count_vis=[count_vis,viscount]
        reg_count_vis=[reg_count_vis,reg_viscount]
    endelse
endfor

end