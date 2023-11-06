;+
;
; NAME:
;   build_count_vis_spectra.pro
;
; PURPOSE:
;     takes in input a standard array of (edited) visibility structures and
;     1) for each energy channel combines them if necessary (hsi_vis_combine.pro)
;     2) for each u,v pair build a count visibility spectrum as a function of the count energy.
;        A spectrum is accepetd only if it is made of 5 points at least.
;     3) save the spectra in the structure "count_vis_spectra"
;
; CALLING SEQUENCE:
;     build_count_vis_spectra, detOK, visib, count_vis_spectra
;
; CALLED BY:
;
;   - visibilities_inversion.pro
;
; CALLS TO:
;
;   - hsi_vis_combine.pro
;
; INPUTS:
;
;   visib:           - Array of count visibility structures.
;   detOK:           - Detectors involved in the inversion
;
; OUTPUTS
;
;   count_vis_spectra:  - A 9-tag structure containing
;   - energyl       dbalrr(neps)            Lower energy of each count bin                  (keV)
;   - energyh       dbalrr(neps)            Upper energy of each count bin                  (keV)
;   - bounds        intarr(2,Nspectra)      For each spectrum, the indices of the first and last energy that compose the spectrum
;   - RealPart      dblarr(neps,Nspectra)   Visibility spectra (Real Part) vs count energy  (cnt/cm^2/s)
;   - ImagPart      dblarr(neps,Nspectra)   Visibility spectra (Imag Part) vs count energy  (cnt/cm^2/s)
;   - error         dblarr(neps,sNspectra)  Errors in visibility values (= for both parts)  (cnt/cm^2/s)
;   - det           intarr(Nspectra)        Detector associated to each spectrum
;   - uv            dblarr(2,Nspectra)      u,v pair associated to each spectrum
;   - accepted      intarr(Nspectra)        1 if the spectrum is accepted and 0 if not (it is accepted only if made of more than 5 samples)
;
;+


pro build_count_vis_spectra, detOK, visib, count_vis_spectra

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
threshold=5     ;;;;; minimum number of samples to build a spectrum for the inversion
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;; Check for the count energy ranges of the visibility bag
epsmin=min(visib.erange[0])
deltaeps=min(visib.erange[1])-epsmin
neps=(max(visib.erange[1])-epsmin)/deltaeps

;;;;;;;; Count energy sampling
eps=epsmin+deltaeps*findgen(neps)

;;;;;;;; For each energy channel combine the visibilities (if necessary)
;;;;;;;; and define peculiarities of the binary table (how many uv pairs
;;;;;;;; for each detector, link between uv-pairs and table columns, ecc)

visptr=ptrarr(neps) ;;;;; array of pointers (one for each energy channel) to visibility structures
uptr=ptrarr(9)   ;;;;; array of pointers (one for each detector) to arrays containing the u values

;;;;;;;;;;;; For the first energy channel [eps[0],eps[1]] :
;;;;;;;;;;;; 1) recall the corresponding visibility structures (and combine if necessary)
;;;;;;;;;;;; 2) for each detector, find the number of the available u,v pairs

indices = WHERE(visib.erange[0] eq eps[0], count)
vis_j=hsi_vis_combine(visib[indices[0:count-1]],/c)
visptr[0]=ptr_new(vis_j)

uv_pairs_per_detector=intarr(9)

FOR isc = 0, 8 DO BEGIN
    indices=WHERE(vis_j.isc eq isc, count)
    if ( count GT 0 ) then begin
        uv_pairs_per_detector[isc]=count
        uptr[isc]=ptr_new(vis_j[indices].u)
    end
END

;;;;;;;;;;;; For the remaining energy channels [eps[1],eps[2]] ..... [eps[neps-2],eps[neps-1]] :
;;;;;;;;;;;; 1) recall the corresponding visibility structures (and combine if necessary)
;;;;;;;;;;;; 2) for each detector, find the number of the available u,v pairs ;
;;;;;;;;;;;;    check if different (u,v) pairs are available with respect to the
;;;;;;;;;;;;    first energy channel comparing the u values and if necessary update the number of
;;;;;;;;;;;;    the uv pairs per detector.

FOR j = 1, neps-1 DO BEGIN

    indices = WHERE(visib.erange[0] eq eps[j], count)
    vis_j=hsi_vis_combine(visib[indices[0:count-1]],/c)
    visptr[j]=ptr_new(vis_j)

    FOR isc = 0, 8 DO BEGIN
       dummy=WHERE(vis_j.isc eq isc, count)
       if ( count Gt 0 ) then begin
        for i = 0, count-1 DO BEGIN
       ;  idiff=where((*uptr[isc]-vis_j[dummy[i]].u) eq 0)
         idiff=where( abs(*uptr[isc]-vis_j[dummy[i]].u) LT 1.e-7 )
         if ( idiff eq -1) then begin
            uv_pairs_per_detector[isc]+=1
            uptr[isc]=ptr_new([*uptr[isc],vis_j[dummy[i]].u])
         endif
        end
       endif

    END
END

;;;;;;;;;;;; For each detector sort the u values in descending order

FOR isc = 0, 8 DO BEGIN
    if (detOK[isc] eq 1) then begin
        usort=*uptr[isc]
        uptr[isc]=ptr_new(usort[REVERSE(SORT(usort))])
    endif
END

;;;;;;;;;;;; The number of spectra to be inverted is = to the number
;;;;;;;;;;;; of available uv pairs.

Nspectra=total(uv_pairs_per_detector,/INTEGER)

count_vis_spectra={energyl:eps[0:neps-1],$
                   energyh:eps[0:neps-1]+deltaeps,$
                   bounds:intarr(2,Nspectra),$
                   RealPart:dblarr(neps,Nspectra),$
                   ImagPart:dblarr(neps,Nspectra),$
                   error:dblarr(neps,Nspectra),$
                   det:intarr(Nspectra),$
                   uv:dblarr(2,Nspectra),$
                   accepted:intarr(Nspectra)}

;first_spectrum_per_detector=total(uv_pairs_per_detector,/cumulative,/preserve_type)-uv_pairs_per_detector

bin_table=intarr(neps,Nspectra)
ispectrum=0
FOR isc = 0, 8 DO BEGIN
  if (detOK[isc] eq 1) then begin
    u=*uptr[isc]
    FOR i = 0, N_elements(u)-1 DO BEGIN
       FOR j = 0, neps-1 DO BEGIN
           vis_j=*visptr[j]
           ind_u=where(vis_j.isc eq isc and abs(vis_j.u-u[i]) LT 1.e-7 ,count)
           if (j eq 0) then v=vis_j[ind_u].v
           bin_table[j,ispectrum]=count
       END

       ;;;;; spectrum with too few samples --> rejected
       if (total(bin_table[*,ispectrum]) LT threshold) then count_vis_spectra.accepted[ispectrum]=0

       ;;;;; spectrum with at least 5 samples --> accepted only if consecutive
       if (total(bin_table[*,ispectrum]) GE threshold) then begin
           j_start=min(where(bin_table[*,ispectrum] eq 1))
           j_stop =min(where(bin_table[j_start+1:neps-1,ispectrum] eq 0))  ;;; first 0 after a serie of 1
           if (j_stop GT 0) then nsamples=j_stop+1 else nsamples=neps-j_start

           if (nsamples LT threshold) then count_vis_spectra.accepted[ispectrum]=0
           if (nsamples GE threshold) then begin
            count_vis_spectra.accepted[ispectrum]=1
            count_vis_spectra.bounds[*,ispectrum]=[j_start,j_start+nsamples-1]
            count_vis_spectra.uv[*,ispectrum]=[u[i],v]
            count_vis_spectra.det[ispectrum]=isc
            FOR j = j_start, nsamples+j_start-1 DO BEGIN
                vis_j=*visptr[j]
                ind_u=where(vis_j.isc eq isc and abs(vis_j.u-u[i]) LT 1.e-7)
                count_vis_spectra.RealPart[j,ispectrum]=float(vis_j[ind_u].obsvis)
                count_vis_spectra.ImagPart[j,ispectrum]=imaginary(vis_j[ind_u].obsvis)
                count_vis_spectra.error[j,ispectrum]=vis_j[ind_u].sigamp
            END
           endif
       endif
       ispectrum=ispectrum+1
    ENDFOR
  endif
ENDFOR

end
