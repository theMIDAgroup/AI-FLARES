function phase_calib_func, xx, extra=extra

rel_gt_factor = extra.rel_gt_factor
rel_dt_factor = extra.rel_dt_factor
rel_dd_factor = extra.rel_dd_factor
cp_factor = extra.cp_factor
subc_ind  = extra.subc_ind
phase     = extra.phase

opt_grid_twist = extra.opt_grid_twist
opt_detector_twist = extra.opt_detector_twist
opt_detector_displ = extra.opt_detector_displ

xx[*, 0] *= opt_grid_twist
xx[*, 1] *= opt_detector_twist
xx[*, 2] *= opt_detector_displ

rel_gt = xx[*, 0] ; REALTIVE GRID TWIST
rel_dt = xx[*, 1] ; RELATIVE DETECTOR TWIST
rel_dd = xx[*, 2] ; RELATIVE DETECTOR DISPLACEMENT

swarmsize = n_elements(rel_gt)
rel_gt = reform(rel_gt, [swarmsize, 1])
rel_dt = reform(rel_dt, [swarmsize, 1])
rel_dd = reform(rel_dd, [swarmsize, 1])

dim  = size(subc_ind, /dim)
if n_elements(dim) eq 1 then begin
n_cp = 1
endif else begin
n_cp = dim[1]
endelse

cp = fltarr(swarmsize, n_cp)

for i = 0, n_cp-1 do begin

cp[*,i] = total((phase[*,subc_ind[*, i]] + rel_gt # reform(rel_gt_factor[subc_ind[*, i]],[1,3]) + $
                                           rel_dt # reform(rel_dt_factor[subc_ind[*, i]],[1,3]) + $
                                           rel_dd # reform(rel_dd_factor[subc_ind[*, i]],[1,3])) * $ 
                                           cp_factor[*,subc_ind[*, i]], 2)

endfor

cp_2pi = atan(sin(cp * !dtor), cos(cp * !dtor)) * !radeg

if n_elements(dim) eq 1 then begin
return, abs(cp_2pi)
endif else begin
return, total(abs(cp_2pi),2)
endelse

end