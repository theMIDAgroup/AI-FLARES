; add the path of the folder that contains the code and the data
folder = '/Users/admin/Documents/PhD/Papers/STIX Imaging group/Github/STIX phase calibration/'

add_path, folder + '/Codes for ps plots/'
add_path, folder + '/Calibration/'

; Restore the data
data = 'phase_19_Nov_2020.dat'
restore, folder + 'Data/' + data, /v

; Correct for a phase shift proportional to the latitude of the source
phase = phase + 26.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; OPTIMIZE CLOSURE PHASES

; Detector labels whose closure phases are chosen for the optimization
det_label = [7,8,9,10]

; Lower and upper bound of the parameters to optimize:
; - grid twist: from -15 to 15 arcmin
; - detector twist: from -15 to 15 arcmin
; - detector displacement: from -1 to 1 mm
lb        = [-15., -15., -1.]
ub        = [15., 15., 1. ]

; Set to 1 the parameters to optimize
opt_grid_twist = 1
opt_detector_twist = 0
opt_detector_displ = 1

n_runs = 100

; Find the parameters that minimize the closure phases
res_str = phase_calib('phase_calib_func', folder + '/Calibration/', lb, ub, phase, det_label, opt_grid_twist = opt_grid_twist, $
                      opt_detector_twist = opt_detector_twist, opt_detector_displ = opt_detector_displ, n_runs=n_runs)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; PLOT BACKPROJECTION LINES

fov      = 512.     ;size of the FOV of the plots (in arcsec)
labels   = [8,9,10] ;labels of the collimators selected for a superimposed plot
                    ;(they must be between 1 and 10)
                    
cf_location = [xy_offset[0],xy_offset[1], 50, 50] ;The first two entries are the x,y coordinates of the flaring source estimated by the CFL
                                                  ;The second two entries are the related uncertainties
                                                  ;(ATTENTION: 50 and 50 are arbitrary values used for this demo! The correct ones should be used)

; Plot the backprojection lines (after phase correction)
stix_bp_lines, res_str.phase_corrected, u / (2.*!pi), v / (2.*!pi), xyoffset = xy_offset, fov = fov, labels = labels, $
               cf_location = cf_location

end