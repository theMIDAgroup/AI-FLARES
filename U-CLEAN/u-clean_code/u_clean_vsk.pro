;+
;
; NAME:
;   u_clean_VSK
;
; PURPOSE:
;   This code computes a VSK interpolant based on the usage of a first rought image reconstruction, 
;   i.e. the clean component map
;
; CALLING SEQUENCE:
;   u_clean_VSK, vis, usampl, N, Ulimit, ep, aux_data, imsize 
;  
; INPUTS:
;   vis: the visibilities 
;   usampl: the evaluation data (i.e. the vector defining the grid)
;   N: number of grid data in one direction
;   Ulimit: the range of the square cointaining the data in the uv-plane
;   ep: the shape parameter for the RBF  
;   aux_data: aux file of the visibilities (mandatory for clean map)
;   imsize: the image size
;
; OUTPUTS:
;   visnew: the visibility surfaces
; 
; NOTES: Part of this code is taken by some Matlab codes available at https://github.com/emmaA89/VSDKs/
;
; HISTORY: Dec 2021, Perracchione E., created
;

function u_clean_vsk, vis, clean_component_map, usampl, N, Ulimit, ep, aux_data, imsize 

;;;;;;;;;;;;;;;;; Store the data in matrices and vectors for interpolation ;;;;;;;;;;;;;;;;;

  vis_matrix = [[(vis.u)], [(vis.v)]]   
  vis_real = normalize_data(float(vis.obsvis))
  vis_imaginary = normalize_data(imaginary(vis.obsvis))
  
;;;;;;;;;;;;;;;;; Define the evaluation points: a matrix (N^2 X 2) ;;;;;;;;;;;;;;;;;

  grid_matrix = [[reform(transpose(cmreplicate(usampl, N)),N^2,1)], $
    [reform(cmreplicate(usampl, N),N^2,1)]]
     
;;;;;;;;;;;;;;;;; Define the regression parameter ;;;;;;;;;;;;;;;;; 
;;;;;;;;;;;;;;;;; It is non-zero only for RHESSI detectors 1 and 2 ;;;;;;;;;;;;;;;;; 
  
  Nall = size(vis_matrix[*,0],/N_ELEMENTS)
  regparam = reform(transpose(0*findgen(Nall[0])),Nall[0])
  wh = [0:size(vis_matrix[*,0],/N_ELEMENTS)-1]
  if (vis[0].type Ne 'stx_visibility') then begin
      if min(vis.isc) LE 1 then begin
         snr_vis = hsi_vis_get_snr(vis)
         wh = where(vis.isc GE 2)
         lambda_reg = 0.01/snr_vis
         regparam[where(vis.isc LE 1)] = lambda_reg
      endif
  endif

;;;;;;;;;;;;;;;;; Define the first approximation of the image ;;;;;;;;;;;;;;;;; 

  if imsize[0] LT 130L then begin
        clean_component_map = make_map(clean_component_map.data,$
          dx=1.,dy=1.)
        newmap = rep_tag_value(clean_component_map, fltarr(imsize),'DATA')
        newmap.dx = 1. & newmap.dy = 1.
        clean_component_map = inter_map(clean_component_map, newmap, err=err)  
  endif else begin
        clean_component_map = make_map(clean_component_map.data,$
          dx=1.,dy=1.)
        newmap = rep_tag_value(clean_component_map, fltarr(imsize),'DATA')
        newmap.dx = 1. & newmap.dy = 1.
        clean_component_map = inter_map(clean_component_map, newmap, err=err)  
  endelse
  if (vis[0].type EQ 'stx_visibility') then begin
        map_dummy = rot_map(make_map(clean_component_map.data),aux_data.ROLL_ANGLE,rcenter=[0.,0.])
        map_coarse =   rotate(map_dummy.data,1)
        px_1 = clean_component_map.dx
        dx = px_1[0]
        dy =  dx
  endif
  
;;;;;;;;;;;;;;;;; Compute the augmented features ;;;;;;;;;;;;;;;;; 
   N_m = sqrt(size(map_coarse,/N_elements))
   augemted_features = u_clean_augmented_feature(map_coarse, Ulimit, grid_matrix, $
    vis_matrix, ep, N_m, dx, dy, aux_data , vis[wh])
  
;;;;;;;;;;;;;;;;; Interpolate the visibilities ;;;;;;;;;;;;;;;;;    
   interp_vis = matern_kernel_interp(augemted_features.vis_matrix_VSK, augemted_features.grid_matrix_VSK, $
    [[vis_real], [vis_imaginary]], ep, regparam = regparam, multi_set = multi_set)

;;;;;;;;;;;;;;;;; Compute the complex grids ;;;;;;;;;;;;;;;;  
   reintz = reform(normalize_data(interp_vis[*,0], second_set = float(vis.obsvis))/$
     (4*!pi*!pi), N, N)  
   imintz = reform(normalize_data(interp_vis[*,1], second_set = imaginary(vis.obsvis))/$
     (4*!pi*!pi), N, N)  
   visnew = rotate(complex(reintz,imintz),4)
    
   return, visnew
  
  end
  
 