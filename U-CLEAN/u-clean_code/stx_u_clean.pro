FUNCTION stx_u_clean, vis,aux_data,clean_component_map,imsize=imsize, pixel=pixel, $
                       ep = ep, flare_loc=flare_loc, $
                       NOPLOT = NOPLOT, uv_window = uv_window, _extra =_extra
 
  ; wrapper around U-CLEAN
  ; output map structure has north up
  
  vis_dupl = stx_vis_duplicate(vis)
 
  u_clean, vis_dupl, clean_component_map, imsize=imsize, pixel=pixel, $
            ep = ep, flare_loc = flare_loc, aux_data = aux_data, u_clean_im, $
            NOPLOT = NOPLOT, uv_window = uv_window, _extra =_extra


  this_method = 'U-CLEAN'
  u_clean_map = stx_make_map(u_clean_im.data, aux_data, pixel, this_method, vis)
  
  return, u_clean_map

END