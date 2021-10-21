;+
;
; NAME:
;   vis_reg_ge_cross3bn.pro
;
; PURPOSE:
;       computes the Singular Value Decomposition of the integral operator associated
;       to Eq.3 in Piana et al. The Astrophysical Journal, 665:846-855, 2007 August 10.
;       This operator is derived from the fully relativistic, solid-angle-averaged,
;       cross section (Koch & Motz 1959, Rev. Mod. Phys.,31, 920-955) including the
;       Elwert collisional correction factor.
;       It optionally includes the term due to electron-electron bremsstrahlung
;
;
; CALLING SEQUENCE:
;
; vis_reg_ge_cross3bn,eebrem,nph,nee,eps,ee,drm,visfit,visfitel,Z,w,u,sf
;
; CALLED BY:
;
;   - vis_regularized_inversion.pro
;   - vis_reg_ge_rescaling.pro
;
; CALLS TO:
;
;   - vis_reg_ge_bremss_cross.pro
;
; INPUTS:
;
;   EEBREM
;      - A number (0 or 1) corresponding to:
;        0 = only Cross3BN
;        1 = Cross3BN plus the term due to electron-electron bremsstrahlung
;
;     NPH
;       - number of count energy bins used
;
;     NEE
;       - number of electron energy bins used
;
;     EPS
;       - count energy vector
;
;      EE
;       - electron energy vector
;
;     DRM
;       - Detector Response Matrix
;
;  VISFIT
;       - Fit of the count visibility spectrum to rescale it
;
;VISFITEL
;       - Rescaling of the electron visibility spectrum
;
;       Z
;       - Value of the root-mean-square atomic number of the target. Default = 1.2
;
; OPTIONAL INPUTS
;   None
;
; KEYWORDS:
;   None
;
; OUTPUTS
;
;    W
;     - singular values of the SVD of the Bremsstrahlung integral operator
;
;    U
;     - singular vectors of the SVD of the Bremsstrahlung integral operator
;
;   SF
;     - singular functions of the SVD of the Bremsstrahlung integral operator
;
; OPTIONAL OUTPUTS:
;   None
;
; COMMON BLOCKS:
;   None.
;
; SIDE EFFECTS:
;   none
;
; RESTRICTIONS:
;
;      We caution the user that addition of the electron-electron bremsstrahlung term
;      increases the computational time considerably.  Unless photon energies significantly
;      above 100 keV are used, the results for 'cross3bn' and 'cross3bnee' are very similar.
;
;-


pro vis_reg_ge_cross3bn,eebrem,nph,nee,eps,ee,drm,visfit,visfitel,Z,w,u,sf

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;; Electron energies sampled starting from the same initial photon energy
;;;;;;;;;; and with the same bin

eebin=ee[1]-ee[0]
epsbin=eps[1]-eps[0]

q=dblarr(nee,nph)

for j=0,nee-1 do begin
    E=ee[j]
    for i=0,nph-1 do begin
       eph=eps[i]
       if (eebrem eq 0) then Qnew=vis_reg_ge_bremss_cross(E,eph,Z,/Noelec)
       if (eebrem eq 1) then Qnew=vis_reg_ge_bremss_cross(E,eph,Z)
       q[j,i]=Qnew
    end
end


counts=eps
Ker=dblarr(nee,nph)

R=1.496e+13 ; 1AU distance in cm
R2pi4=4.*!PI*R^2

for iq=0,nph-1 do begin
  for j=0,nee-1 do begin
    Ker(j,iq)=0.
    for i=0,nph-1 do begin
       if( (eps[i] ge counts[iq]) and (eps[i] le ee[j])) then Ker[j,iq]= Ker[j,iq]+drm[iq,i]*q[j,i]*epsbin
    end
    Ker[j,iq]=1.d50*eebin*Ker[j,iq]/R2pi4
   end
end

;;;;;;;;;; rescaling


    for iq=0,nph-1 do begin
      for j=0,nee-1 do begin
           Ker[j,iq]=Ker[j,iq]*visfitel[j]/visfit[iq]
     endfor
    endfor


;;;;;;;;;;;;;;;;;;;;

svdc,Ker,w,u,v,/double

sf=v    ;;;; singular functions

end
