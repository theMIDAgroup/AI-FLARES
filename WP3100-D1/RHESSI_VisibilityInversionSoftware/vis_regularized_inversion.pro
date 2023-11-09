;+
;
; NAME:
;   vis_regularized_inversion.pro
;
; PURPOSE:
;     for a given (real or imaginary) count visibility spectrum, performs the regularized inversion as follows:
;     1) check the consistency of the input vectors
;     2) rescale the input count visibility spectrum
;     3) invert the count visibility spectrum to provide regularized electron visibility spectrum according to Eq.3
;        in Piana et al. The Astrophysical Journal, 665:846-855, 2007 August 10
;     4) repeat the inversion several times in order to provide a confidence strip around the regularized electron visibility spectrum
;     5) save the inverted spectra in a structure
;
; CALLING SEQUENCE:
;     vis_regularized_inversion, l, part, count_vis_spectra, datastruct, drmini, electron_bin=electron_bin, confidencestrip=confidencestrip
;
; CALLED BY:
;   - visibilities_inversion.pro
;
; CALLS TO:
;   - vis_reg_ge_sampling.pro
;   - vis_reg_ge_cross3bn.pro
;   - vis_reg_ge_regularization.pro
;   - vis_reg_ge_confidence.pro
;
; INPUTS:
;   l:                 - Label of the spectrum (l=0, .., Total number of spectra to invert -1)
;   part:              - 'Real' or 'Imaginary'
;   count_vis_spectra: - The 9-tag structure, output of the build_count_vis_spectra.pro code, containing
;                        the information on the count visibility spectra
;   drmini:            - The drm computed by compute_drm.pro
;
; KEYWORDS:
;   electron_bin       - Width (keV) of the energy bins in the returned electron visibility array (=.width of the count bins)
;   confidencestrip    - Number of solution realizations to determine. (Default = 10)
;
; OUTPUTS
;
;   DATASTRUCT - A structure storing 2 arrays
;     - Tag 1 :     STRIP       DOUBLE    Array[NEE, N+2]
;
;       where NEE = number of electron energy points
;               N = number of solution realizations (defined by confidencestrip value)
;
;       The array contains the electron energies and each realization of the regularized solution
;       in the following format:
;
;       datastruct.strip[*,0]     =  a vector of NEE elements containing the electron energies
;       datastruct.strip[*,1]     =  a vector of NEE elements containing the original (unperturbed) regularized
;                                    solution at the specified energies
;       datastruct.strip[*,2:N+1] =  array containing N different regularized electron visibilities, at the
;                                    specified energies, produced by randomly perturbing the input data
;
;     - Tag 2 : RESIDUALS       DOUBLE    Array[NPH, 6]
;
;       where NPH = number of count energy points
;
;       The data is in six columns, with each row arranged as follows:
;
;       datastruct.residuals[*,0] = Count energy used
;       datastruct.residuals[*,1] = Count visibility values (input data)
;       datastruct.residuals[*,2] = Uncertainty in count visibility values (input data)
;       datastruct.residuals[*,3] = Count visibilities corresponding to the recovered electron visibility array
;       datastruct.residuals[*,4] = Residual count visibilities (actual - recovered, i.e., column 2 - column 4)
;       datastruct.residuals[*,5] = Cumulative residual, defined as
;                                   C_j = (1/j) sum_[i=1]^j res_i
;
;+

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro vis_regularized_inversion, l, part, opt, count_vis_spectra, datastruct, drmini, $
    electron_bin=electron_bin,                   $ ; Width (keV) of the energy bins in the returned
                                                   ; electron flux array. (Range [1,5] - default=1)
    confidencestrip=confidencestrip                ; Number of solution realizations to determine. (Range [1,50] -
                                                   ; default = 10).


DEFAULT, photon_bin_position,   0.5
DEFAULT, el_energy_max_factor,  2
DEFAULT, electron_bin,          1
DEFAULT, crosssection,          'Cross3BN'
DEFAULT, Z,                     1.2

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

if (part eq 'Real') then vis_array=count_vis_spectra.RealPart[*,l] else vis_array=count_vis_spectra.ImagPart[*,l]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;; Some cheking;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;  Testing vector sizes and contents

n=N_elements(count_vis_spectra.energyl)
if ( (n ne N_elements(count_vis_spectra.energyh)) or $
     (n ne N_elements(vis_array)) or $
     (n ne N_elements(count_vis_spectra.error[*,l])) ) then message, 'INPUT VECTOR SIZES NOT COMPATIBLE'

if ( (total(count_vis_spectra.energyl) eq 0) or $
     (total(count_vis_spectra.energyh) eq 0) or $
     (total(vis_array) eq 0) or $
     (total(count_vis_spectra.error[*,l]) eq 0)) then message, 'INPUT VECTOR CONTENTS ARE ZEROS'

if ( where(vis_array eq 'Inf') ne -1 )  then message, 'Inf values in the visibility array'
if ( where(count_vis_spectra.error[*,l] eq 'Inf') ne -1 )  then message, 'Inf values in the visibility array'

;;;;;;;;;;  Testing sample number

if( n le 3) then begin
    datastruct={strip:0,residuals:0}
    return
endif


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;; The code automatically fits the photon spectrum with a single, broken
;;;;;;;;;;; power law or sum of power laws. Then it rescales the spectrum.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

vis_reg_ge_rescaling,count_vis_spectra.energyl, count_vis_spectra.energyh,$
    vis_array,count_vis_spectra.error[*,l],drmini,electron_bin,$
    el_energy_max_factor,nph,eps,gs,err,wei,fitph,fitel,drm

if( nph le 3) then begin
    datastruct={strip:0,residuals:0}
    return
endif
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,;
;;;;;;;;;;; Electron energies sampling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

eemax=el_energy_max_factor*eps[nph-1]

if (electron_bin gt (eemax-eps[0])/(nph-1)) then electron_bin=(eemax-eps[0])/(nph-1)

nee=ceil((eemax-eps[0])/electron_bin+1)
ee=findgen(nee)*electron_bin+min(eps)

;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;; Cross-section computation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

eebrem=0
vis_reg_ge_cross3bn,eebrem,nph,nee,eps,ee,drm,fitph,fitel,Z,w,u,sf

;;;;;;;;; Rescaling of the singular vectors and
;;;;;;;;; values according to the weights
for k=0,nph-1 do begin
    w[k]=sqrt(wei[k])*w[k]
    u[*,k]= u[*,k]/sqrt(wei[k])
end
;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;; Regularized solution computation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;rescaling
gs=gs/fitph
err=err/fitph
;;;;;;;;;;;;;;;;;;;;

vis_reg_ge_regularization,nph,nee,wei,eps,gs,err,u,w,sf,fitph,fitel,$
    ee,opt,gopt,regsol,residual_array, cumulativeresidual_array

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;; Confidence strip computation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

vis_reg_ge_confidence,nph,nee,ee,regsol,$
    err,gs,eps,wei,confidencestrip,u,w,opt,sf,drm,fitph,fitel,Z,datastruct

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;  Fill the fourth tag of datastruct structure and save it ;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


datastruct.residuals[*,0]=eps[*]
datastruct.residuals[*,1]=gs[*]
datastruct.residuals[*,2]=err[*]
datastruct.residuals[*,3]=gopt[*]
datastruct.residuals[*,4]=residual_array[*]
datastruct.residuals[*,5]=cumulativeresidual_array[*]

end