;+
;
; NAME:
;   rbf_vsk
;
; PURPOSE:
;   This code computes a VSK interpolant based on the usage of a first rought image reconstruction
;
; CALLING SEQUENCE:
;   rbf_vsk, vis, lambda_reg, usampl, N, method, Ulimit, p, imsize, fov_zp, fov, pixel_xy
;
; INPUTS:
;   vis: the visibilities 
;   lambda_reg: regression parameter
;   usampl: the evaluation data (i.e. the vector defining the grid)
;   N: number of grid data in one direction
;   method: the method used to initialize the procedure (either back-projection or clean component map)
;   Ulimit: the range of the square cointaining the data in the uv-plane
;   p: treshold used to segment the back-projection map
;   imsize: size of the output image
;   fov: fov in the uv-plane
;   fov_zp: fov after the zero padding strategy
;   pixel_xy: size of the output pixel
;
; OUTPUTS:
;   visnew: the visibility surfaces
; 
; NOTES: Part of this code is taken by some Matlab codes available at https://github.com/emmaA89/VSDKs/

function rbf_vsk, vis, lambda_reg, usampl, N, method, Ulimit, p, imsize, fov_zp, fov, pixel_xy

  dsites = ([[vis.u], [vis.v]])

  indin= []
  indout =[]
  for i=0L,size(dsites,/N_ELEMENTS)/2-1 do begin
    if sqrt((dsites[i,0])^2.+(dsites[i,1])^2.) ge 0.08 then begin
      indout = [indout,i];
    endif else begin
      indin = [indin,i];
    endelse
  end
  datain = dsites[indin,*]
  dataout = dsites[indout,*]
  ; Reorder and rescale data
  
  dsites = [datain,dataout]
  dsites = (dsites+Ulimit)/(2*Ulimit)

  ; Reorder and rescale the function values
  rhsreal = float(vis.obsvis)
  rhsim = imaginary(vis.obsvis)
  rhsrealin = rhsreal[indin]
  rhsrealout = rhsreal[indout]
  rhsimin = rhsim[indin]
  rhsimout = rhsim[indout]
  rhsrealt = [rhsrealin,rhsrealout]
  rhsimt = [rhsimin,rhsimout]
  rhsReal = (rhsrealt-min(rhsrealt))/max(rhsrealt-min(rhsrealt))
  rhsIm = (rhsimt-min(rhsimt))/max(rhsimt-min(rhsimt))

  ; Define the regression parameter
  nn1 = size(datain,/N_ELEMENTS)/2
  Nall = size(dsites,/N_ELEMENTS)/2
  regparam = reform(transpose(0*findgen(Nall[0])),Nall[0])
  if nn1 le Nall[0]-1 then regparam[nn1:-1] = lambda_reg
     epoints = data_preprocess(usampl, Ulimit,N)
 
   if (method eq 'BP') then begin
       vis_bpmap, vis,  BP_FOV=imsize, PIXEL=1., MAP=map
       map_fa = MAP
   endif
   if (method eq 'CC') then begin
     clean_im = vis_clean(vis, niter = 100, gain = 0.05, image_dim = imsize, $
       pixel = 1., clean_sources_map = clean_sources_map, noresid = 1)
     map_fa = CONGRID(clean_sources_map, 128, 128)
   endif
   
   ; Segment the rought map 
   tres = MAKE_ARRAY(imsize, imsize, /FLOAT, VALUE = p*max(max(map_fa)))
   new_mappa =  (map_fa)*(abs(map_fa) ge tres)
    
   ; Compute the FFT of the sought map to define the VSK scale function
   pixel_psi = fov/imsize
   Ulimit_psi=(imsize/2.-1)*pixel_psi+pixel_psi/2.
   xpixnew=-Ulimit_psi+findgen(imsize)*pixel_psi
   deltaomega=(xpixnew[imsize-1]-xpixnew[0])/imsize
   
   IFTTPSI = shift(new_mappa,[-imsize/2.,-imsize/2.])
   FTTPSI = FFT(IFTTPSI,/inverse)/(4*!pi*!pi*deltaomega*deltaomega*imsize*imsize)
   PSI_ALL = shift(FTTPSI,[imsize/2.,imsize/2.])
   
   
   ; Reduce the map to those grid data needed by uv-smooth
 
if (pixel_xy eq 1.) then begin
   N_map = N
endif
if (pixel_xy eq 0.5) then begin
   N_map = 2*N
endif 

PSI = rotate(PSI_ALL(imsize/2-floor(((N_map*imsize)/fov_zp)/2):imsize/2+floor(((N_map*imsize)/fov_zp)/2),$
  imsize/2-floor(((N_map*imsize)/fov_zp)/2):imsize/2+floor(((N_map*imsize)/fov_zp)/2)),4)
PSI = rotate(PSI_ALL(imsize/2-floor(((N_map*imsize)/fov_zp)/2):imsize/2+floor(((N_map*imsize)/fov_zp)/2),$
  imsize/2-floor(((N_map*imsize)/fov_zp)/2):imsize/2+floor(((N_map*imsize)/fov_zp)/2)),4)   

; Define the shape parameter for the RBF interpolants and compute the augmented features
ep = 0.01
data_add = augmented_feature(PSI, epoints, dsites, ep, nn1, Nall)
dsites_VSK = data_add.dsites_vsk
epoints_VSK = data_add.epoints_vsk
 
; Compute the kernel matrix for interpolation
DM = distance_matrix(dsites_VSK,dsites_VSK)
IM = exp(-ep*DM)
   
;; Solve the linear systems
coefi = reform(LA_LINEAR_EQUATION(IM+diag_matrix(transpose(regparam)), transpose(rhsIm)),size(dsites,/N_ELEMENTS)/2,1)
coefr = reform(LA_LINEAR_EQUATION(IM+diag_matrix(transpose(regparam)), transpose(rhsReal)),size(dsites,/N_ELEMENTS)/2,1)

   ; Evaluate the models
   if Ulimit ge 0.15 then begin
     imintz = MAKE_ARRAY(size(epoints,/N_ELEMENTS)/2, 1,  /FLOAT, VALUE = 0)
     reintz = MAKE_ARRAY(size(epoints,/N_ELEMENTS)/2, 1,  /FLOAT, VALUE = 0)
     for i=0,size(epoints,/N_ELEMENTS)/2-1 do begin
       DM_eval = distance_matrix(epoints_VSK[i,*],dsites_VSK);
       EM = exp(-ep*DM_eval)
       imintz[i] = (EM#coefi)*max(imaginary(vis.obsvis)-min(imaginary(vis.obsvis)))+min(imaginary(vis.obsvis))
       reintz[i] = (EM#coefr)*max(float(vis.obsvis)-min(float(vis.obsvis)))+min(float(vis.obsvis))
     endfor
   endif else begin
   DM_eval = distance_matrix(epoints_VSK,dsites_VSK)
   EM = exp(-ep*DM_eval)
   ; Rescale and define the data
   reintz = (EM#coefr)*max(float(vis.obsvis)-min(float(vis.obsvis)))+min(float(vis.obsvis))
   imintz = (EM#coefi)*max(imaginary(vis.obsvis)-min(imaginary(vis.obsvis)))+min(imaginary(vis.obsvis))
   endelse
   
   reintz = (reform(reintz/(4*!pi*!pi), N, N))
   imintz = (reform(imintz/(4*!pi*!pi), N, N))
   visnew = rotate(complex(reintz,imintz),4)
   return, visnew
  
  end