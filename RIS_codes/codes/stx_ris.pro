FUNCTION stx_RIS, vis_tot, aux_data, trigger_map, imsize=imsize, pixel=pixel, $
                       ep = ep, flare_loc=flare_loc, $
                       NOPLOT = NOPLOT, uv_window = uv_window, _extra =_extra
 
  ;************ Selected energy intervals to perform the RIS approach ************
  lower_energy_edge = vis_tot[uniq(vis_tot.energy_range[0], sort(vis_tot.energy_range[0]))].energy_range[0]
  upper_energy_edge = vis_tot[uniq(vis_tot.energy_range[1], sort(vis_tot.energy_range[1]))].energy_range[1]
  
  RIS_maps = [] ; Empty array of maps
  
  for i = 0, n_elements(lower_energy_edge)-1 do begin
    energy_range = [lower_energy_edge[i],upper_energy_edge[i]] ;selected energy range
    vis = vis_tot[where(vis_tot.energy_range[0] eq lower_energy_edge[i])]
    vis_dupl = stx_vis_duplicate(vis)
    
    ; The first map is obtained by using as scaling funcion the dicrete Fourier trasform of the triggering map
    if i eq 0 then begin
      print, ''
      print, 'Energy range'
      print, energy_range
      print, ''
      int_extr_routine, vis_dupl, trigger_map, imsize=imsize, pixel=pixel, $
        ep = ep, flare_loc = flare_loc, aux_data = aux_data, RIS_im, $
        NOPLOT = NOPLOT, uv_window = uv_window, _extra =_extra
      this_method = 'RIS-method'
      RIS_map = stx_make_map(RIS_im.data, aux_data, pixel, this_method, vis)
      RIS_maps = [RIS_maps, RIS_map]
    endif else begin
      print, ''
      print, 'Energy range'
      print, energy_range
      print, ''
      ; Each map is obtained by using as scaling funcion the dicrete Fourier trasform of the map obtained at the previous energy channel
      int_extr_routine, vis_dupl, RIS_map, imsize=imsize, pixel=pixel, $
        ep = ep, flare_loc = flare_loc, aux_data = aux_data, RIS_im, $
        NOPLOT = NOPLOT, uv_window = uv_window, _extra =_extra
      this_method = 'RIS-method'
      RIS_map = stx_make_map(RIS_im.data, aux_data, pixel, this_method, vis)
      RIS_maps = [RIS_maps, RIS_map]
    endelse
    
  endfor
  return, RIS_maps

END