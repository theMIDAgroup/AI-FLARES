add_path, 'C:\Users\volpa\OneDrive - unige.it\Desktop\Dottorato\Codici_uv_smooth_12Dicembre2022\prova_energie\RIS_codes\codes\'

; UID of the science fits file to be dowloaded from the website
uid_sci_file = "2211113411"
path_sci_file = stx_get_science_fits_file(uid_sci_file, out_dir=out_dir)
; Time range to be selected for image reconstruction
time_Range = ['11-Nov-2022 01:30:00', '11-Nov-2022 01:32:00']
; Download L2 ephemeris fits file 
aux_fits_file = stx_get_ephemeris_file(time_range[0], time_range[1], out_dir=out_dir)
; Construct auxiliary data structure
aux_data = stx_create_auxiliary_data(aux_fits_file, time_range)

subc_index = stx_label2ind(['10a','10b','10c','9a','9b','9c','8a','8b','8c','7a','7b','7c',$
  '6a','6b','6c','5a','5b','5c','4a','4b','4c','3a','3b','3c'])
stx_estimate_flare_location, path_sci_file, time_range, aux_data, flare_loc=flare_loc
mapcenter_stix = stx_hpc2stx_coord(flare_loc, aux_data)
xy_flare_stix  = mapcenter_stix

; Set image and pixel size
imsize = [128, 128] ; Number of pixels of the map to be reconstructed
pixel  = [1, 1]     ; Pixel size in arcsec

; Triggering visibilities and triggering map
vis = stx_construct_calibrated_visibility(path_sci_file, time_range, [4.,6.],$
  mapcenter_stix, subc_index=subc_index, path_bkg_file=path_bkg_file, xy_flare=xy_flare_stix)
mem_ge_map = stx_mem_ge(vis,imsize,pixel,aux_data)

stop

; Create an array of STIX count visibility structures
lower_energy_edge = [6.,8.,10.,12.,14.,16.,18.,20.,22.]
upper_energy_edge = [8.,10.,12.,14.,16.,18.,20.,22.,25.]
vis_tot = []                                                   ;empty array of visibilities
for kk=0, n_elements(lower_energy_edge)-1 do begin
  energy_range = [lower_energy_edge[kk],upper_energy_edge[kk]] ; selected energy range
  this_estring_0=strtrim(fix(energy_range[0]),2)
  this_estring_1=strtrim(fix(energy_range[1]),2)
  vis = stx_construct_calibrated_visibility(path_sci_file, time_range, energy_range,$
    mapcenter_stix, subc_index=subc_index, path_bkg_file=path_bkg_file, xy_flare=xy_flare_stix)
  vis_tot = [vis_tot, vis]
endfor

stop

;******************************* RIS APPROACH *******************************
; The regularized imaging spectroscopy approach can be formulated according to the following scheme:
;
; 1. Given a set of visibilities corresponding to a starting energy channel (e.g., ε0=[4., 6.]):
;    a. Use any visibility-based image reconstruction method (e.g., MEM_GE) to compute the triggering 
;       map for this energy channel.
; 2. Apply Fourier transform to compute the scale function associated with the triggering map obtained 
;    in Step 1.
; 3. Given the visibility set in an energy channel ϵ1 adjacent to ϵ0, apply VSK interpolation with the 
;    scale function obtained in Step 2 to interpolate visibilities in the spatial frequency domain.
; 4. Use a constrained iterative method to reconstruct the count map at energy ε1.
; 5. Repeat Steps 3–4 iteratively for each energy channel up to the final one (εI).

; Comments:
; - vis_tot is the visibility array in the energy intervals where the RIS method is to be applied 
;   NOTE: the energy channel triggering the process (ε0) is NOT be included in this array
; - mem_ge_map is the triggering map 
; - Ris_maps_out is an array of maps obtained by using the RIS approach

Ris_maps_out = stx_RIS(vis_tot, aux_data, mem_ge_map, imsize=imsize, pixel=pixel, flare_loc = xy_flare_stix)

stop

window, /free, xsize = 1500., ysize = 800.
!p.multi = [0,5,2]
plot_map, mem_ge_map,title =  'MEM_GE 4-6 keV', /limb, grid_spacing =2.5,charsize=2., fov = 2., $
  gcolor=255, lcolor = 255., gthick = 0.3

for i=0,n_elements(lower_energy_edge)-1 do begin
  energy_range = [lower_energy_edge[i],upper_energy_edge[i]]
  this_estring_0=strtrim(fix(energy_range[0]),2)
  this_estring_1=strtrim(fix(energy_range[1]),2)
  plot_map, Ris_maps_out[i],title =  this_estring_0+'-'+this_estring_1+'keV', /limb, grid_spacing =2.5,charsize=2., fov = 2., $
    gcolor=255, lcolor = 255., gthick = 0.3
end


end