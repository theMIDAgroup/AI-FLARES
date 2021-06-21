
;+
;
; NAME:
;   uv_smooth_rbfvsk
;
; PURPOSE:
;   This code implements a new interpolation routine for uv-smooth:
;   - utilizes RHESSI or STIX visibilities as input
;   - replaces the unknown visibilities at non-sampled (u,v) points with a VSK interpolant
;   - utilizes an iterative routine incorporating a positivity constraint on the image to provide
;     visibilities that taper gradually to zero outside the sampling domain.
;
; CALLING SEQUENCE:
;   uv_smooth_vsk, vis, lambda_reg, method, p,
;
; INPUTS:
;   vis:  input visibility structure in standard format
;   lambda_reg: regression parameter
;   method: the method used to initialize the procedure (either back-projection or clean component map)
;   p: a treshold used to segment the back-projection map
;
; OUTPUTS:
;   map: image map in the structure format provided by the routine make_map.pro
; KEYWORDS:
;   NOPLOT - default to 1 to not plot uv plane of available visibilities
; RESTRICTIONS:
;   - input visibilities must be sampled in the whole (u,v) plane (visibilities
;     with only positive v values must be previuosly converted to their conjugate values)
;
; MODIFICATION HISTORY:
;   Dec-2008 Written Anna M. Massone
;   Jul-2009, ras, added RECONSTRUCTED_MAP_VISIBILITIES as an output
;       NOPLOT added to prevent uncontrolled graphics
;       REMOVE_LIST - list of detector isc's not to use because they aren't implemented yet
;   16-Jul-2009, Kim.  Added uv_window arg. Reuse that window for vis sampling plot
;   17-Nov-2009, Anna, Kim. Removed remove_list arg and logic. Can now handle all 9 detectors.
;   26-Nov-2011, Kim. Call al_legend instead of legend (IDL V8 conflict)
;-

pro uv_smooth_rbfvsk, vis, lambda_reg, method, p, map, reconstructed_map_visibilities = F_trasf, $
  NOPLOT=NOPLOT, uv_window=uv_window, _extra=_extra

  default, noplot, 1
  default, uv_window, -1

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  mapcenter=vis[0].xyoffset
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  Rmax=max(sqrt(vis.u*vis.u+vis.v*vis.v))
  
  pixel=0.0005
       
  ;;;;; Check if the visibilities are from RHESSI or STIX

  if (vis[0].type Eq 'stx_visibility') then begin
     fov=0.16
     pixel_xy = 1.
     N=320L
  endif else begin    
    ;vis = vis0[where_arr( vis0.isc, remove_list, /notequal)] ;clean the list
    detmin=min(vis.isc)
    
    ;Rmin=min(sqrt(vis.u*vis.u+vis.v*vis.v))
    ;if (Rmax GT 0.13) then begin
    ;    message, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',/continue
    ;    message, 'Detector 1 must not be used',/continue
    ;    message, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    ;endif
    
    ;;;;;;;; Do not change these numbers!

    if (detmin eq 0 ) then begin
      fov=0.45
      pixel=pixel*2.
      pixel_xy = 0.5
      N=450L
    endif
    if (detmin eq 1 ) then begin
      fov=0.26
      pixel=pixel*2.
      pixel_xy = 0.5
      N=260L
    endif
    if (detmin GE 2 ) then begin
      fov=0.16
      N=320L
      pixel_xy = 1.
      endif
    endelse

  ;;;;;;;;;  Definition of the uniform grid for the interpolation
  Ulimit=(N/2.-1)*pixel+pixel/2.
  usampl=-Ulimit+findgen(N)*pixel
  vsampl=usampl
    
  ;;;;;;;;;  Plot of the visibility sampling on the (u,v)-plane    
  A = FINDGEN(17) *  (!PI*2/16.)
  if not NOPLOT then begin
    ; Save current window. Reuse old uv_window if available, otherwise create it.
    save_window=!d.window
    if uv_window eq -1 or (is_wopen(uv_window) eq 0) then begin
      uv_window = next_window(/user)
      window, uv_window, xsize=500, ysize=500,xpos=0,ypos=50,title='u-v sampling '
    endif
    wset, uv_window
    USERSYM, 1./3.*COS(A), 1./3.*SIN(A), /FILL
    plot, vis.u, vis.v,  /isotropic, psym=8, xtitle='u (arcsec!u-1!n)', ytitle='v (arcsec!u-1!n)', $
      title='Visibility Sampling in u-v plane'
    if (vis[0].type Eq 'stx_visibility') then begin
    aux = vis[0].time_range[0].value
    time = anytim(aux.time, mjd = aux.mjd, /ecs)
    tr = minmax(time)
    al_legend, [format_intervals(tr,/ut)], box=0
    endif else begin  
    tr = minmax(vis.trange)
    det = 'Detectors: ' + arr2str(trim(get_uniq(vis.isc)+1), ' ')
    al_legend, [format_intervals(tr,/ut), det], box=0
    endelse
    ; Reset to user's window
    wset,save_window
  endif
  
  ;;;;;;;;;  Fix the following parameters
  imsize = 128L
  Nnew = 1920L
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Interpolation with VSKs
  visnew = rbf_vsk(vis, lambda_reg, usampl, N, method, Ulimit, p, imsize, Nnew, fov, pixel_xy)
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Do not allow function to extrapolate outside the disk
  for i=0L,N-1 do begin
    for j=0L,N-1 do begin
      if( sqrt(usampl[i]*usampl[i]+vsampl[j]*vsampl[j]) GT Rmax ) then visnew[i,j]=complex(0.,0.)
    end
  end

  ;;;;;;;;;;;;;;;;; Arrange parameter values before applying the FFT  ;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;; First step: zero-padding (Do not change these numbers!)
  
  Ulimit=(Nnew/2.-1)*pixel+pixel/2.
  xpix=-Ulimit+findgen(Nnew)*pixel
  ypix=xpix

  intzpadd=MAKE_ARRAY(Nnew,Nnew, /complex, VALUE = 0)
  intzpadd[(Nnew-N)/2.:(Nnew-N)/2.+N-1,(Nnew-N)/2.:(Nnew-N)/2.+N-1]=visnew
    
  ;;;;;;;;;;;;;;;; Second step: resampling of the image with a mask 15*15
  ;;;;;;;;;;;;;;;; (Do not change these numbers!)
  im_new=Nnew/15.
  intznew=complexarr(im_new,im_new)
  xpixnew=fltarr(im_new)
  ypixnew=fltarr(im_new)

  for i=0,im_new-1 do begin
    xpixnew[i]=xpix[15*i]
    ypixnew[i]=ypix[15*i]
    for j=0,im_new-1 do intznew[i,j]= intzpadd[15*i,15*j]
  end

  ;;;;;;;;;;; Compute the fov and the sampling distance in the space-space
  ;;;;;;;;;;; given the fov and sampling distance in the frequency-space
  OMEGA=(xpixnew[im_new-1]-xpixnew[0])/2.
  X=im_new/(4*OMEGA)
  deltaomega=(xpixnew[im_new-1]-xpixnew[0])/im_new
  deltax=2*X/im_new

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;; FFT computation
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;; store the interpolated/resampled visibilities ;;;;;;;;;
  g=intznew
  ;;;;;;;;;;;;; Visibilities are defined as the inverse Fourier Transform of the flux ('+' in their definition)
  ;;;;;;;;;;;;; --> to recover the flux we have to compute the Fourier Transform of the visibilities
  intznew=shift(intznew,[-im_new/2.,-im_new/2.])
  fftInverse =4*!pi*!pi*deltaomega*deltaomega*im_new*im_new*float(FFT(intznew))
  fftInverse=shift(fftInverse,[im_new/2.,im_new/2.])

  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;;; Projected Landweber method with positivity constraint
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;; Characteristic function of the disk

  chi=complexarr(im_new,im_new)
  for i=0,im_new-1 do begin
    for j=0,im_new-1 do begin
      if (sqrt(xpixnew[i]*xpixnew[i]+ypixnew[j]*ypixnew[j]) LE Rmax) then chi[i,j]=complex(1.,0.)
    end
  end

  ;;;;;;;;;;;;;;;;;;;  Landweber iterations

  iterLand=50 ;;;;;;; maximum number of Landweber iterations

  map_actual=complexarr(im_new,im_new)
  map_iteration=fltarr(iterLand,im_new,im_new)
  map_solution=fltarr(im_new,im_new)

  tau=0.2 ;;;;;;; relaxation parameter

  descent=fltarr(iterLand-1)
  normAf_g=fltarr(iterLand)

  ;;;;;;;;;;;;;;; iteration 0: Inverse Fourier Transform of the initial solution
  map_shifted=shift(map_actual,[-im_new/2.,-im_new/2.])
  F_Trasf_shifted =FFT(map_shifted,/inverse)/(4*!pi*!pi*deltaomega*deltaomega*im_new*im_new)
  F_Trasf=shift(F_Trasf_shifted,[im_new/2.,im_new/2.])

  ;;;;;;;;;;;;;;; Landweber iterations

  for iter=0,iterLand-1 do begin

    ;;;; Landweber updating rule:
    F_Trasf_up=F_Trasf+ tau*(g- chi*F_Trasf)
    F_Trasf=F_Trasf_up
    
    ;;;; Fourier Transform of the updated solution
    F_Trasf_shifted=shift(F_Trasf,[-im_new/2.,-im_new/2.])
    map_shifted=FFT(F_Trasf_shifted)*4*!pi*!pi*deltaomega*deltaomega*im_new*im_new
    map_actual=shift(map_shifted,[im_new/2.,im_new/2.])

    ;;;; Projection of the solution onto the subset of the positive solutions (positivity constraint)
    for i=0,im_new-1 do begin
      minzero=where( float(map_actual[0:im_new-1,i]) lt 0,count)
      if (count NE 0) then  map_actual[minzero,i] =complex(0.,0.)
    endfor

    map_iteration[iter,*,*]=float(map_actual)

    map_shifted=shift(map_actual,[-im_new/2.,-im_new/2.])
    F_Trasf_shifted =FFT(map_shifted,/inverse)/(4*!pi*!pi*deltaomega*deltaomega*im_new*im_new)
    F_Trasf=shift(F_Trasf_shifted,[im_new/2.,im_new/2.])

    ;;;;;;;;;;;;;;;; Stop criterion based on the descent of ||Af-g||

    Af_g=chi*F_Trasf-g
    normAf_g[iter]=sqrt(total(abs(Af_g)*abs(Af_g)))
    if (iter GE 1) then begin
      descent[iter-1]=(normAf_g[iter-1]-normAf_g[iter])/normAf_g[iter-1]
      if (descent[iter-1] LT 0.02  ) then break
    endif
  endfor

  F_Trasf=4*!pi*!pi*F_Trasf ;F_TRASF are the final interpolated and reconstructed visibilities after positivity constraint

  ;;;;;;;;;;;;;;;;;;; Map corresponding to the optimal iteration
  if (iter eq iterLand) then map_solution[*,*]=map_iteration[14,*,*]  else map_solution=float(map_actual)

  ;;;;;;;;;;;;;;;;;;;  Image dimension: 128x128;  pixel size = about 1 arcsec  
  
  flag=1
  catch, error_status

  if error_status ne 0 then begin
    aux = vis[0].time_range[0].value
    time = anytim(aux.time, mjd = aux.mjd, /ecs)
    flag = 0
    catch, /cancel
  endif

  if flag then time = anytim(vis[0].trange[0], /ecs)

  map=make_map(map_solution,$
    id=' ', $ ;earth orbit
    xc=mapcenter[0],yc=mapcenter[1],$
    dx=deltax,dy=deltax,$
    time=time, $
    xunits='arcsec', yunits='arcsec')
    
end