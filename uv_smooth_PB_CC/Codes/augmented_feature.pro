;+
;
; NAME:
;   augmented_feature
;
; PURPOSE:
;   This code computes the added features found out via the function PSI
;
; CALLING SEQUENCE:
;   augmented_feature, PSI,  epoints, dsites, ep, nn1, Nall
;   ;
; INPUTS:
;   PSI: the VSK map
;   epoints: the evaluation data 
;   dsites: the data sites or nodes (i.e. a grid)
;   ep: the shape parameter of the kernel
;   nn1: number of points out of the det3 for RHESSI
;   Nall: number of data
;
; OUTPUTS:
;   data_add: the added coordinates for both data and evaluation points
;

function augmented_feature, PSI,  epoints, dsites, ep, nn1, Nall
  
  ; Define the data sites, e.g. the grid of the map PSI
  N1 = size(PSI, /dimensions)
  usampl_PSI = (findgen(N1[0]))/((N1[0]-1))
  yy = replicate1(usampl_PSI, N1[0])
  xx = transpose(yy)
  dsitesx = reform(xx,(N1[0])^2,1)
  dsitesy = reform(yy,(N1[0])^2,1)
  dsites_PSI = [[dsitesx], [dsitesy]]

  ; Compute the kernel matrix 
  DM = distance_matrix(dsites_PSI,dsites_PSI)
  IM = exp(-ep*DM)
  
  ; Solve the systems 
  rhsr = float(reform(PSI,N1[0]^2,1))
  coefr = reform(LA_LINEAR_EQUATION(IM, rhsr[*,0]),N1[0]^2,1)
  rhsi =  imaginary(reform(PSI,N1[0]^2,1))
  coefi = reform(LA_LINEAR_EQUATION(IM, rhsi[*,0]),N1[0]^2,1)
  
  ; Evaluate the models
  if nn1 le Nall[0]-1 then begin
  epointsadd_real = reform(CONGRID(float(PSI), sqrt(size(epoints,/N_ELEMENTS)/2), $
    sqrt(size(epoints,/N_ELEMENTS)/2)),size(epoints,/N_ELEMENTS)/2,1)
  epointsadd_imag = reform(CONGRID(imaginary(PSI), sqrt(size(epoints,/N_ELEMENTS)/2), $
    sqrt(size(epoints,/N_ELEMENTS)/2)),size(epoints,/N_ELEMENTS)/2,1)
    
  R_d=max(sqrt((dsites[0:nn1-1,0]-0.5)*(dsites[0:nn1-1,0]-0.5)+(dsites[0:nn1-1,1]-0.5)*(dsites[0:nn1-1,1]-0.5)))

  DM_eval = distance_matrix(dsites,dsites_PSI);
  EM = exp(-ep*DM_eval)
  dsitesadd_imag = (EM)#(coefi);
  dsitesadd_real = (EM)#(coefr);
      
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Do not allow function to extrapolate outside the disk
for i=0L,(size(epoints,/N_ELEMENTS)/2)-1 do begin
    if(sqrt((epoints[i,0]-0.5)*(epoints[i,0]-0.5)+(epoints[i,1]-0.5)*(epoints[i,1]-0.5))) GT 2*R_d then begin
       epointsadd_real[i]=0.
       epointsadd_imag[i]=0.
    endif
end
for i=0L,Nall-1 do begin
if(sqrt((dsites[i,0]-0.5)*(dsites[i,0]-0.5)+(dsites[i,1]-0.5)*(dsites[i,1]-0.5))) GT 2*R_d then begin
      dsitesadd_real[i]=0.
      dsitesadd_imag[i]=0.
 endif
 end 
 endif else begin
  DM_eval = distance_matrix(epoints,dsites_PSI)
  EM = exp(-ep*DM_eval)
  epointsadd_imag = (EM)#(coefi)
  epointsadd_real = (EM)#(coefr)
  DM_eval = distance_matrix(dsites,dsites_PSI)
  EM = exp(-ep*DM_eval)
  dsitesadd_imag = (EM)#(coefi)
  dsitesadd_real = (EM)#(coefr)
endelse

   
  ; Append those data to the original ones and rescale
  epointsvsk = [[epoints], [(epointsadd_real-min(epointsadd_real))$
    /max(epointsadd_real-min(epointsadd_real))], [(epointsadd_imag-min(epointsadd_imag))$
    /max(epointsadd_imag-min(epointsadd_imag))]];
  dsitesvsk = [[dsites], [(dsitesadd_real-min(dsitesadd_real))$
    /max(dsitesadd_real-min(dsitesadd_real))], [(dsitesadd_imag-min(dsitesadd_imag))$
    /max(dsitesadd_imag-min(dsitesadd_imag))]];

  ; Return the output
  data_add = {struct, name:'', dsites_vsk:dsitesvsk, epoints_vsk:epointsvsk}
  return, data_add
 
end