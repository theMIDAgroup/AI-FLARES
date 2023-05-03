add_path, '~/Documenti/idl873-linux/Codici3Maggio_2023/uv_smooth_codes/'

data_folder    = '~/Documenti/idl873-linux/Codici17Gennaio_2023/Dati/'
outfolder_plots = '~/Documenti/idl873-linux/Codici3Maggio_2023/Immagini/'
;.compile hsi_image_vis_plot_1, distance_matrix, matern_kernel_interp, normalize_data, replicate_array, stx_map2pixelabcd_matrix, uv_smooth, uv_smooth_augmented_feature, uv_smooth_vsk,  vis_duplicate


;;;************************************************ 9-May-2021 14:00:55 ************************************************
path_sci_file = data_folder + $
  'solo_L1A_stix-sci-xray-l1-2105090024_20210509T134653-20210509T140055_011210_V01.fits'
path_bkg_file = data_folder + $
  'solo_L1A_stix-sci-xray-l1-2105080012_20210508T020005-20210508T034005_010936_V01.fits'
time_range = ['9-May-2021 13:53:00', '9-May-2021 13:55:00']


subc_index = stx_label2ind(['1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c','5a','5b','5c','6a','6b','6c',$
  '7a','7b','7c','8a','8b','8c','9a','9b','9c','10a','10b','10c'])

out_dir = concat_dir( getenv('stx_demo_data'),'imaging', /d)

pixel = [1,1]
imsize = [128L,128L]


path_sci = data_folder
path_sci_file = [path_sci + 'solo_L1A_stix-sci-xray-l1-2108260030_20210826T231549-20210826T232115_013330_V01.fits']

time_range = ['26-Aug-2021 23:18:00', '26-Aug-2021 23:20:00']

xy_flare =  [1053.653, -770.691]
mapcenter = xy_flare

aux_fits_file = path_sci + 'solo_L2_stix-aux-ephemeris_20210826_V01.fits'

subc_index3 = stx_label2ind(['10a','10b','10c','9a','9b','9c','8a','8b','8c','7a','7b','7c',$
  '6a','6b','6c','5a','5b','5c','4a','4b','4c','3a','3b','3c', '2a','2b','2c']);, '1a','1b','1c'])

  energy_Range = [15.,25.]

  boh = time_range
  aux_data = stx_create_auxiliary_data(aux_fits_file, time_range)
  mapcenter_stix = stx_hpc2stx_coord(mapcenter, aux_data)
  xy_flare_stix  = stx_hpc2stx_coord(xy_flare, aux_data)
  mapcenter_stix = mapcenter_stix 
  xy_flare_stix = xy_flare_stix 
  vis3 = stx_construct_calibrated_visibility(path_sci_file, time_range, energy_Range, mapcenter_stix, $
    path_bkg_file=path_bkg_file, xy_flare=xy_flare_stix, subc_index=subc_index3)

  vis_bis = vis_duplicate(vis3)
  uv_smooth_map_bp = stx_uv_smooth(vis_bis,aux_data,imsize=imsize, pixel=pixel,method='BP', threshold_PSI=0.85,  ep = 0.1, flare_loc = xy_flare_stix)
  uv_smooth_map_cc = stx_uv_smooth(vis_bis,aux_data,imsize=imsize, pixel=pixel,method='CC', threshold_PSI=0., ep = 0.1, flare_loc = xy_flare_stix)
  
  
  xsize = 1700
  ysize = 1200
  loadct, 5, /silent
  ;hsi_linecolors
  !p.multi = [0,1,2]
  set_plot, 'z'
  device, set_resolution=[xsize, ysize]
  loadct, 5, /silent
  plot_map, uv_smooth_map_bp,  title = + 'UV_SMOOT_BP ' + boh[0]+ ' '  + num2str(energy_Range[0])+ ' ' + num2str(energy_Range[1])+' Kev' , charsize=1.5 , /limb
  plot_map, uv_smooth_map_cc,  title = + 'UV_SMOOT_CC ' + boh[0]+ ' '  + num2str(energy_Range[0])+ ' ' + num2str(energy_Range[1])+' Kev', charsize=1.5,  /limb
  thisimage = tvrd()
  tvlct, r, g, b, /get
  image24 = bytarr(3, xsize, ysize)
  image24(0,*,*) = r(thisimage)
  image24(1,*,*) = g(thisimage)
  image24(2,*,*) = b(thisimage)
  thisfile_full = outfolder_plots + 'mappe' + '_' + num2str(shift_mappa[0])+ '_' + boh[0]+'.jpg'
  write_jpeg, thisfile_full, image24, true=1
  set_plot, 'X'


  xsize1=1201
  ysize1=501
  loadct, 5, /silent
  ;hsi_linecolors
  set_plot, 'z'
  device, set_resolution=[xsize1, ysize1]
  loadct, 5, /silent
  stx_plot_fit_map, uv_smooth_map_bp, title = + 'UV_SMOOT_BP', chi2=chi2, this_window=kk+5
  thisimage = tvrd()
  tvlct, r, g, b, /get
  image24 = bytarr(3, xsize1, ysize1)
  image24(0,*,*) = r(thisimage)
  image24(1,*,*) = g(thisimage)
  image24(2,*,*) = b(thisimage)
  thisfile_full = outfolder_plots + 'Chi_UV_SMOOT_BP' + '_' + num2str(shift_mappa[0])+ '_' + boh[0]+ '.jpg'
  write_jpeg, thisfile_full, image24, true=1
  set_plot, 'X'
  
  
  xsize2=1202
  ysize2=502
  loadct, 5, /silent
  ;hsi_linecolors
  set_plot, 'z'
  device, set_resolution=[xsize2, ysize2]
  loadct, 5, /silent
  stx_plot_fit_map, uv_smooth_map_cc, title = + 'UV_SMOOT_CC', chi2=chi2, this_window=kk+6
  thisimage = tvrd()
  tvlct, r, g, b, /get
  image24 = bytarr(3, xsize2, ysize2)
  image24(0,*,*) = r(thisimage)
  image24(1,*,*) = g(thisimage)
  image24(2,*,*) = b(thisimage)
  thisfile_full = outfolder_plots + 'Chi_UV_SMOOT_CC' + '_' + num2str(shift_mappa[0])+ '_' + boh[0]+ '.jpg'
  write_jpeg, thisfile_full, image24, true=1
  set_plot, 'X'

end
