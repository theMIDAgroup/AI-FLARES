
;;;;;;;;;;;;;;;; Add the path of the folder containing the code
folder      = '/Users/admin/Documents/PhD/Papers/STIX Imaging group/Github/STIX imaging from amplitudes/'
code_folder = folder + 'Code/'
add_path, code_folder

;;;;;;;;;;;;;;;; Restore the data
data_folder = folder + 'Data/'
energy_min  = 26
restore, data_folder + 'data_18Nov2020_054530_054615_' + num2str(fix(energy_min)) + '_70keV.dat'

;;;;;;;;;;;;;;;; Detector used for the reconstruction (1a,1b,1c,2a,2b,2c are excluded)
subc_str = stx_construct_subcollimator()

det_idx = where(subc_str.label ne 'cfl' and subc_str.label ne 'bkg')
ddet_idx = where(subc_str.label ne 'cfl' and $
  subc_str.label ne 'bkg' and $
  subc_str.label ne '1a' and $
  subc_str.label ne '1b' and $
  subc_str.label ne '1c' and $
  subc_str.label ne '2a' and $
  subc_str.label ne '2b' and $
  subc_str.label ne '2c')

;;;;;;;;;;;;;;;; Type of the parametric shape used in the forward fit:
; - 'circle': single circular Gaussian source
; - 'ellipse': single elliptical Gaussian source
; - 'multi': two circular Gaussian sources
type = 'multi'

;;;;;;;;;;;;;;;; Set to 1 for the uncertainty estimation of the parameters (confidence strip approach)
uncertainty = 0

;;;;;;;;;;;;;;;; Degrees of freedom
det_used = num2str(n_elements(ddet_idx))
if type eq 'circle' then n_free = det_used - 2.
if type eq 'ellipse' then n_free = det_used - 4.
if (type eq 'multi') or (type eq 'multi_fl') then n_free = det_used - 6.

;;;;;;;;;;;;;;;; Visibility amplitudes
ampobs = data.amp_both[ddet_idx]

;;;;;;;;;;;;;;;; Error on the visibility amplitudes. 5% of systematic error is added
sigamp = data.amp_both_error[ddet_idx]
syserr = 0.05
sigamp = SQRT(sigamp^2  + syserr^2 * ampobs^2)

;;;;;;;;;;;;;;;; u,v coordinates of the angular frequencies sampled by STIX (nominal values). Needed in the forward fit
dummy_vis = stx_construct_visibility(subc_str[ddet_idx])
u = dummy_vis.u
v = dummy_vis.v

;;;;;;;;;;;;;;;; FORWARD FIT WITH PSO
result = amp_fwdfit(type,'amp_fwdfit_func', ampobs, sigamp, u, v, n_free, uncertainty = uncertainty, Swarmsize = 100)
; - 'result.srcstr' contains the reconstructed parameters
; - 'result.fitsigmas' contains the estimated uncertainties

;;;;;;;;;;;;;;;; Plot the reconstructed map
imsize = [128, 128]
pixel = [1., 1.]
amp_fwdfit_source2map, result.srcstr, [0., 0.], amp_map, pixel = pixel, mapsize = imsize
amp_fwdfit_map = make_map(amp_map, xcen = 0., ycen = 0., dx = pixel[0], dy = pixel[1], id = 'STIX from amplitudes')

loadct, 5
window, 0
plot_map, amp_fwdfit_map, /cbar, title = 'PSO reconstruction'


;;;;;;;;;;;;;;;; Compute the chi2 value of the reconstruction
ampobs = data.amp_both[det_idx]
sigamp = data.amp_both_error[det_idx]
syserr = 0.05
sigamp = SQRT(sigamp^2  + syserr^2 * ampobs^2)
dummy_vis = stx_construct_visibility(subc_str)
u = dummy_vis.u
v = dummy_vis.v

F = vis_map2vis_matrix(u, v, imsize, pixel)
vispred = F # amp_fwdfit_map.data[*]
ampobsmap = abs(vispred)

ind_d = [26L, 6L, 0L,  $
  4L, 22L, 20L, $
  27L, 5L, 1L,  $
  12L, 28L, 24L, $
  21L, 25L, 7L, $
  18L, 3L, 23L, $
  29L, 11L, 13L,$
  19L, 17L, 2L]

chi2 = total((ampobsmap[ind_d] - ampobs[ind_d])^2./sigamp[ind_d]^2.)/n_free

;;;;;;;;;;;;;;;; Plot the fit of the visibility amplitudes
xx = (findgen(30))/3. + 1.2
xx = xx[6:29]

linecolors, /quiet

charsize = 1.2
leg_size = 1.4
thick = 1.6

title = 'Visibilities - Observed, From Image - CHI2: ' + num2str(chi2)
units = 'counts s!U-1!n keV!U-1!n'
xtitle = 'Label'

window, 1

plot, xx, ampobs[ind_d], /nodata, xrange=[1.,11.], /xst, xtickinterval=1, xminor=-1, $
  title=title, xtitle=xtitle, ytitle=units, yrange=yrange, charsize=charsize, thick=thick, _extra=_extra

; draw vertical dotted lines at each detector boundary
for i=1,10 do oplot, i+[0,0], !y.crange, linestyle=1


errplot, xx, (ampobs[ind_d]-sigamp[ind_d] > !y.crange[0]), ampobs[ind_d]+sigamp[ind_d] < !y.crange[1], $
  width=0, thick=thick, COLOR=7
oplot, xx, ampobs[ind_d], psym=7, thick=thick
oplot, xx, ampobsmap[ind_d], psym=4, col=2, thick=thick

leg_text = ['Observed', 'Error on Observed', 'From Image']
leg_color = [255, 7,2]
leg_style = [0, 0, 0]
leg_sym = [7, -3, 4]
ssw_legend, leg_text, psym=leg_sym, color=leg_color, linest=leg_style, box=0, charsize=leg_size, thick=thick, /left

  


end