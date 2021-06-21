;+
;
; NAME:
;   stix_bp_lines
;   
; PURPOSE:
;   Procedure for plotting the backprojection lines
;
; CALLING SEQUENCE:
;   stix_bp_lines, phase, u, v
;
; INPUTS:
;   phase: array with the 30 visibility phases (in degrees)
;   u: array with the 30 u coordinates of the sampled frequencies 
;   v: array with the 30 v coordinates of the sampled frequencies
;
; KEYWORDS:
;   xyoffset: array containing the coordinates of the center of the plots (in arcsec). Default, [0,0]
;   fov: size of the field of view of the plots (in arcsec). Default, 128
;   labels: array containing the labels (between 1 and 10) of the collimators selected for a superimposed plot.
;           If not passed, the plot is not displayed
;   map: map structure. If passed, the lines corresponding to the collimators selected in labels are plotted over the map
;   ps_folder: sring containing the path of a folder. If passed, the plots are saved as .ps files in that folder
;   cf_location: array of four elements. The first two entries are the x,y coordinates (in arcsec) of the flaring source estimated
;                by the CFL, the second ones are the related uncertainties. If set, the position estimated by the CFL
;                is overplotted as a circle and the uncertainties are plotted as the semi axes of an ellipse around the position 
;
; HISTORY: April 2021, Massa P., Perracchione E. and Garbarino S. created 
;
; CONTACT:
;   massa.p [at] dima.unige.it
;-

pro stix_bp_lines, phase, u, v, xyoffset=xyoffset, fov=fov, labels = labels, map = map, ps_folder = ps_folder, cf_location=cf_location

  default, xyoffset, [0., 0.]
  default, fov, 128.
  
  ; If a map is passed, change the coordinates of the center and the FOV to be consistent
  if keyword_set(map) then begin
  
  xyoffset = [map.xc, map.yc]
  dim = size(map.data, /dim)
  fov = dim[0] * map.dx
  
  endif
  
  phase = phase / 180. * !pi ; phase in radians
  
  ; Discretization of the x-axis (used in the plots)
  xx = (findgen(257)/256 - 0.5)*fov + xyoffset[0]

  ; Detector indices (between 0 and 29)
  det_ind = [[9,11,16], $
    [10,17,15], $
    [7,27,1], $
    [23,5,21], $
    [6,28,2], $
    [13,25,29], $
    [22,8,26], $
    [19,24,4], $
    [14,12,30], $
    [3,18,20]]-1

  ; Resolution of the collimators
  res30=1./(2.*sqrt(u^2. + v^2.))
  ; Grid label (between 1 and 10)
  grid_label = indgen(10)+1

  loadct, 5
  
  ; Define the circle symbol for plots
  A = FIndGen(16) * (!PI*2/16.)
  UserSym, cos(A), sin(A), /fill
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;;;;;;;;;;;; Plot all collimators ;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  thick=1.
  if keyword_set(ps_folder) then begin
  
  charsize = 1.1
  thick = 1.5
 
  print_options, /land
  popen, ps_folder + '/bp_all_collimators.ps', units='cm', xsize=10.8, ysize=5
  
  endif else begin
    
  charsize = 2.0
  window, 0, xsize=1250,ysize=600
  
  endelse
  
  !p.multi = [0, 5, 2]
  
  for h =0, 9 do begin ; For loop on the collimators label

  this_title= 'Label: ' + num2str(fix(grid_label(h)))
  plot, xx, -u[det_ind[0, h]]/v[det_ind[0, h]]*xx + phase[det_ind[0, h]]/(2.*!pi*v[det_ind[0, h]]), /xst, $
        yrange = [xyoffset[1] - fov/2., xyoffset[1] + fov/2.], /yst, /nodata, /isotropic, charsize=charsize, thick=thick, $
        xtitle = 'X (arcsec)', ytitle = 'Y (arcsec)', title = this_title 

  for j=0, 2 do begin ; For loop on the three collimators with the same angular resolution
  
  ; Computation of the number of lines needed for covering the FOV
  if -u[det_ind[j, h]]/v[det_ind[j, h]] ge 0 then begin
    tmp = round(v[det_ind[j, h]] * (xyoffset[1] - fov/2. + u[det_ind[j, h]]/v[det_ind[j, h]] * (xyoffset[0] + fov/2)) $
      - phase[det_ind[j, h]]/(2.*!pi))
  endif else begin
    tmp = round(v[det_ind[j, h]] * (xyoffset[1] - fov/2. + u[det_ind[j, h]]/v[det_ind[j, h]] * (xyoffset[0] - fov/2)) $
      - phase[det_ind[j, h]]/(2.*!pi))
  endelse
  n_k = round(fov * sqrt(2)/res30[det_ind[j, h]]/2) + 1
  k = signum(v[det_ind[j, h]]) * findgen(n_k) + tmp

  for i=0, n_k-1 do begin ; For loop for plotting the lines

  oplot, xx, -u[det_ind[j, h]]/v[det_ind[j, h]]*xx + phase[det_ind[j, h]]/(2.*!pi*v[det_ind[j, h]]) + k[i]/v[det_ind[j, h]], thick=thick

  endfor
  endfor
  
  if keyword_set(cf_location) then begin
  oplot, [cf_location[0]], [cf_location[1]], psym = 8, color = 125 ; Plot the position esimated by the CFL
  TVELLIPSE, cf_location[2], cf_location[3], cf_location[0], cf_location[1], color=125, thick=thick, /DATA
  endif
  
  endfor

  xyouts,0.5,1-0.12/2.5,'Backprojection lines',/normal,chars=1.6,ali=0.5
  
  if keyword_set(ps_folder) then pclose

  !p.multi = 0
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;;;;;;;;; Plot selected collimators ;;;;;;;;;;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  if keyword_set(labels) then begin
    
  if keyword_set(ps_folder) then begin

  charsize = 1.1
 
  print_options, /port
  popen, ps_folder + '/bp_selected_collimators.ps', units='cm', xsize=7, ysize=6
  
  endif else begin

  charsize = 1.6
  window, 1, xsize=720,ysize=600

  endelse
  
  labels = labels - 1
  n_labels = n_elements(labels)
  this_title = 'Label: ' + strjoin(STRSPLIT(string(grid_label[labels], /print), /EXTRACT), ',')

  
  if keyword_set(map) then begin
    
  plot_map, map, /cbar, charsize = charsize
    
  endif else begin
  
  set_viewport,0.15,0.8,0.15,0.9

  plot, xx, -u[det_ind[0, labels[0]]]/v[det_ind[0, labels[0]]]*xx + phase[det_ind[0, labels[0]]]/(2.*!pi*v[det_ind[0, labels[0]]]), $
    /xst, yrange = [xyoffset[1] - fov/2., xyoffset[1] + fov/2.], /yst, /nodata, /isotropic, title = this_title, charsize=charsize, $
    xtitle = 'X (arcsec)', ytitle = 'Y (arcsec)', thick=thick

  endelse

  for h =0, n_labels-1 do begin ; For loop on the collimators label
  for j=0, 2 do begin ; For loop on the three collimators with the same angular resolution
  
  ; Computation of the number of lines needed for covering the FOV
  if -u[det_ind[j, labels[h]]]/v[det_ind[j, labels[h]]] ge 0 then begin
    tmp = round(v[det_ind[j, labels[h]]] * (xyoffset[1] - fov/2. + u[det_ind[j, labels[h]]]/v[det_ind[j, labels[h]]] * (xyoffset[0] + fov/2)) $
      - phase[det_ind[j, labels[h]]]/(2.*!pi))
  endif else begin
    tmp = round(v[det_ind[j, labels[h]]] * (xyoffset[1] - fov/2. + u[det_ind[j, labels[h]]]/v[det_ind[j, labels[h]]] * (xyoffset[0] - fov/2)) $
      - phase[det_ind[j, labels[h]]]/(2.*!pi))
  endelse
  n_k = round(fov * sqrt(2)/res30[det_ind[j, labels[h]]]/2) + 1
  k = signum(v[det_ind[j, labels[h]]]) * findgen(n_k) + tmp
 

  for i=0, n_k-1 do begin ; For loop for plotting the lines
  
  if keyword_set(ps_folder) then begin
  oplot, xx, -u[det_ind[j, labels[h]]]/v[det_ind[j, labels[h]]]*xx + phase[det_ind[j, labels[h]]]/(2.*!pi*v[det_ind[j, labels[h]]]) $
    + k[i]/v[det_ind[j, labels[h]]], color = 255/(n_labels+2)*h, thick=thick  
  endif else begin
  oplot, xx, -u[det_ind[j, labels[h]]]/v[det_ind[j, labels[h]]]*xx + phase[det_ind[j, labels[h]]]/(2.*!pi*v[det_ind[j, labels[h]]]) $
    + k[i]/v[det_ind[j, labels[h]]], color = 255 - 255/(n_labels+2)*h, thick=thick
  endelse

  endfor
  endfor
  
  ; Plot of the legend
  xyouts,0.83,0.21, 'Legend: ', /normal, chars=charsize,ali=0.5, orientation = 90
  if keyword_set(ps_folder) then begin
 
  if h le 4 then begin
  xyouts,0.83,0.21 + (h+1)*0.12, 'label ' + num2str(fix(grid_label[labels[h]])), /normal, chars=charsize,ali=0.5, $
    color = 255/(n_labels+2)*h, orientation = 90
  endif else begin
  xyouts,0.86,0.21 + (h-4)*0.12, 'label ' + num2str(fix(grid_label[labels[h]])), /normal, chars=charsize,ali=0.5, $
    color = 255/(n_labels+2)*h, orientation = 90
  endelse
  
  endif else begin
    
  if h le 4 then begin
  xyouts,0.83,0.21 + (h+1)*0.12, 'label ' + num2str(fix(grid_label[labels[h]])), /normal, chars=charsize,ali=0.5, $
    color = 255-255/(n_labels+2)*h, orientation = 90
  endif else begin
  xyouts,0.86,0.21 + (h-4)*0.12, 'label ' + num2str(fix(grid_label[labels[h]])), /normal, chars=charsize,ali=0.5, $
    color = 255-255/(n_labels+2)*h, orientation = 90
  endelse
  endelse

  endfor
  
  if keyword_set(cf_location) then begin
  if keyword_set(ps_folder) then begin
  oplot, [cf_location[0]], [cf_location[1]], psym = 8, color = 255/(n_labels+2)*(h+1)
  TVELLIPSE, cf_location[2], cf_location[3], cf_location[0], cf_location[1], color=255/(n_labels+2)*(h+1), thick=2.*thick, /DATA
  
  if h le 4 then begin
    xyouts,0.83,0.21 + (h+1)*0.12, 'CFL', /normal, chars=charsize,ali=0.5, $
      color = 255/(n_labels+2)*(h+1), orientation = 90
  endif else begin
    xyouts,0.86,0.21 + (h-4)*0.12, 'CFL', /normal, chars=charsize,ali=0.5, $
      color = 255/(n_labels+2)*(h+1), orientation = 90
  endelse
  
  endif else begin
  oplot, [cf_location[0]], [cf_location[1]], psym = 8, color = 255-255/(n_labels+2)*(h+1)
  TVELLIPSE, cf_location[2], cf_location[3], cf_location[0], cf_location[1], color=255-255/(n_labels+2)*(h+1), thick=2.*thick, /DATA
  
  if h le 4 then begin
    xyouts,0.83,0.21 + (h+1)*0.12, 'CFL', /normal, chars=charsize,ali=0.5, $
      color = 255-255/(n_labels+2)*(h+1), orientation = 90
  endif else begin
    xyouts,0.86,0.21 + (h-4)*0.12, 'CFL', /normal, chars=charsize,ali=0.5, $
      color = 255-255/(n_labels+2)*(h+1), orientation = 90
  endelse
  
  endelse
  endif


  if keyword_set(ps_folder) then pclose
  if ~keyword_set(map) then !p.position = [0, 0, 0, 0]
  
  endif


end