;;;;;;;;;; Define the folder and compile the routines
folder = '~/Documenti/idl873-linux/TestPaperEmma/TestStix/uv_smooth_PB_CC/'
;CD, folder + 'Codes/'
;.compile data_preprocess, distance_matrix, augmented_feature, replicate1, rbf_vsk, uv_smooth_rbfvsk  

;;;;;;;;;;; Restore data
;;;;;;;;;;; Visibilities must be dublicate. They can be STIX or RHESSI data
;;;;;;;;;;; Here visibilities are STIX simulated data
restore, folder + 'Data/' + 'vis.sav'

;;;;;;;;;;; Define the inputs for uv_smooth_rbfvsk
method = 'BP'
;;;;;;;;;;; if method = BP, uv_smooth_rbfvsk uses the feature augmentation strategy via BP map
;;;;;;;;;;; if method = BP, uv_smooth_rbfvsk uses the feature augmentation strategy via Clean Component Map
p = 0.9 
;;;;;;;;;;; p is a parameter for thresholding the BP map or the Clean Component Map
;;;;;;;;;;; if method = BP then 0.7<=p<=0.9
;;;;;;;;;;; if method = CC then 0<=p<=0.3
lambda_reg = 0.
;;;;;;;;;;; lambda_reg is the regression parameter. Set lambda_reg=0. unless RHESSI detectors 1-2 are used

;;;;;;;;;;; Reconstruct the flare image via uv_smooth_rbfvsk
uv_smooth_rbfvsk, vis_bis, lambda_reg, method, p, uv_VSK_im

;;;;;;;;;;; Plot the map
xsize=820
ysize=850
csize = 2.
loadct, 5, /silent
hsi_linecolors
set_plot, 'z'
device, set_resolution=[xsize, ysize]
plot_map, uv_VSK_im, /cbar, /no_timestamp,  /equal, XTITLE='x (arcsec)', YTITLE ='y (arcsec)', $
  bottom=13, title ='', cb_position=[.16, .95, .95, .97], charsize=csize, position=[.16, .1, .95, .85]
thisimage = tvrd()
tvlct, r, g, b, /get
image24 = bytarr(3, xsize, ysize)
image24(0,*,*) = r(thisimage)
image24(1,*,*) = g(thisimage)
image24(2,*,*) = b(thisimage)
thisfile_full = folder + 'ResutsPlots/' + 'VSK_map.jpg'
write_jpeg, thisfile_full, image24, true=1
set_plot, 'X'

end