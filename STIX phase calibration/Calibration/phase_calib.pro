function phase_calib, fun_name, folder, lb, ub, phase, det_label, SwarmSize = SwarmSize, TolFun = TolFun, maxiter = maxiter, $
  opt_grid_twist = opt_grid_twist, opt_detector_twist = opt_detector_twist, opt_detector_displ = opt_detector_displ , $
  n_runs = n_runs, silent = silent

  default, SwarmSize, 100.
  default, TolFun, 1e-06
  default, opt_grid_twist, 1
  default, opt_detector_twist, 1
  default, opt_detector_displ, 1
  default, n_runs, 1
  default, silent, 0
  
  ;;;;;;; Detector indices (between 0 and 29)
  det_ind = [[9,11,16], $ ; Label 1
            [10,17,15], $ ; Label 2
            [7,27,1], $   ; Label 3
            [23,5,21], $  ; Label 4
            [6,28,2], $   ; Label 5
            [13,25,29], $ ; Label 6
            [22,8,26], $  ; Label 7
            [19,24,4], $  ; Label 8
            [14,12,30], $ ; Label 9
            [3,18,20]]-1  ; Label 10
            
  subc_ind = det_ind[*, det_label-1]
  
  ;;;;;;;;;;;;;;;;;;;;;; PHASE CALIBRATION FACTORS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  calib_factors = read_csv(folder + 'PhaseCalibrationFactors.csv', header=header, table_header=tableheader)
  det_ind  = calib_factors.field1
  sort_ind = sort(det_ind)
  det_ind = det_ind[sort_ind]
  
  ; CLOSURE PHASE FACTOR
  cp_factor = calib_factors.field2
  cp_factor = cp_factor[sort_ind]
  
  ; RELATIVE GRID TWIST FACTOR
  rel_gt_factor = calib_factors.field3
  rel_gt_factor = rel_gt_factor[sort_ind]

  ; RELATIVE DETECTOR TWIST FACTOR
  rel_dt_factor = calib_factors.field4
  rel_dt_factor = rel_dt_factor[sort_ind]

  ; RELATIVE DETECTOR DISPLACEMENT FACTOR
  rel_dd_factor = calib_factors.field5
  rel_dd_factor = rel_dd_factor[sort_ind]

  phase_copy = transpose(cmreplicate(phase, SwarmSize))
  cp_factor_copy = transpose(cmreplicate(cp_factor, SwarmSize))

  extra = {phase:phase_copy, $
    subc_ind: subc_ind, $
    cp_factor: cp_factor_copy, $
    rel_gt_factor: rel_gt_factor, $
    rel_dt_factor: rel_dt_factor, $
    rel_dd_factor: rel_dd_factor, $
    opt_grid_twist: opt_grid_twist, $
    opt_detector_twist: opt_detector_twist, $
    opt_detector_displ: opt_detector_displ}


  Nvars = n_elements(lb)
  if ~keyword_set(maxiter) then maxiter = Nvars*SwarmSize

  xopt = fltarr(n_runs, 3)
  fopt = fltarr(n_runs)
  for i=0, n_runs-1 do begin
  optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
  xopt[i, *] = optim_f.xopt
  fopt[i] = optim_f.fopt
  endfor
  
  min_fopt = min(fopt, ind_min)

  phase_corrected = phase + xopt[ind_min, 0] * rel_gt_factor + xopt[ind_min, 1] * rel_dt_factor + xopt[ind_min, 2] * rel_dd_factor
  
  if ~keyword_set(silent) then begin
  
  print, " "
  print, "Relative grid twist: " + num2str(xopt[ind_min, 0]) + " arcmin"
  print, "Relative detector twist: " + num2str(xopt[ind_min, 1]) + " arcmin"
  print, "Relative detector displacement: " + num2str(xopt[ind_min, 2]) + " mm"
  print, " "
  print, "Sum closure phases: " + num2str(fopt[ind_min])
  print, " "
  
  endif

  return, {rel_gt: xopt[ind_min, 0], $
           rel_dt: xopt[ind_min, 1], $
           rel_dd: xopt[ind_min, 2], $
           phase_corrected: phase_corrected}

end