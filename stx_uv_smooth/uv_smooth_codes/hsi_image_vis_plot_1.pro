;+
; Name: hsi_image_vis_plot
;
; Purpose: Plot the observed visibilities and the visibilities computed from an image
;
; Method: Call hsi_image_vis_chi2 to get the observed and image visibilitied and compare them -
;  it returns the chisquare values as well as the vis amps and all values for plotting.
;  Plots the amplitude of the observed visibilities and their errors for each position angle for
;  each detector (even those that weren't used in image). Overplots the amplitude of the image
;  visibilities. Plot is sent to screen or PS file. Chi-square values for comparison of two
;  sets of visibilities for all detectors, all detectors used in image, and each detector
;  separately, are displayed on plot and can be returned in keyword argument.
;
;
; Input Keywords:
;  imobj - image object
;
; Optional Input Keywords:
;  Note: now that calculations are done in separate routine hsi_image_vis_chi2, many of these keywords are simply
;  passed through to that routine.
;  image - if set, use this image instead of calling getdata on image object (still need image object to correspond to
;    image, since will be getting other values like detectors, pixel_size, xyoffset, etc from object)
;  tb_index - Time index to plot (defaults to 0)
;  eb_index - Energy index to plot (defaults to 0)
;  last - if set, uses highest tb_index and eb_index available in image object
;  force - if set, and didn't pass in image, and image object needs update, will remake image without asking
;  visfile - if visibilities aren't available from image object without reprocessing, and visfile is supplied, get vis from file
;  ps - if set, send output to PS file
;  jpeg - If set, send plot to jpeg file
;  file_plot - name of plot file to use. Default is to autoname to
;    'viscomp_alg_prefix_t0_e0-e1kev.xx' where alg_prefix is short prefix name for image algorithm, t0 is start time of image,
;    and e0,e1 are start end of energy bin.
;  dir_plot - directory to use for output plot file (default is viscomp_plot_dir parameter, or if that's blank, current dir)
;  window - window number to use (default is whatever is in viscomp_window parameter)
;  yrange - yrange of plot, (defaults to range of last 20 elements of obsvis amplitudes to avoid bad values on finer collimators)
;  no_plot - if set, do all the work, but don't make the plot (useful for returning chival without plotting, NOTE; kept
;  this keyword for backward compatibility, but could now call hsi_image_vis_chi2 directly to get chival)
;  _extra - any keywords to pass to plot program
;  quiet - default is 1, set quiet=0 to see the vis combine/edit/normalize messages
;
; Output Keywords:
;  chivals - structure containing reduced chi-square for all vis and for each detector of comparison between observed and expected
;    visibilities.  Computed by total( (expx-obsx)^2 + (expy - obsy)^2) / obssig^2 ) / (np - 1), where expx, expy are the x and y
;    values of the expected vis (from image), obsx, obsy are the observed vis, obssig is sigma in amplitude of observed vis, and
;    np is the number of points used.
;
; Written: Kim Tolbert Jan. 2017
; Modifications:
; 3-Jul-2017, Kim. Even if vis are not already made, if we have image obj with a single time and energy, calib eventlist
;   is still available, so use that to make vis. (previously if vis not made, always made new obj and made vis from scratch).
;   Also, right side of red box showing which dets were used in image didn't show for Det 9, so make it 9.99 instead of 10.
;   Also, for reg vis, conjugates are always combined, so can't use value of vis_conj flag to control max angle to plot.
; 14-Sep-2017, Kim. Fixed red chi2 calc to divide by 2*nvis since using x and y components separately. Modified chi2 labels
;   on plot to make clearer and indicate that it's red chi2, not full. Added 'albedo removed' label if vf_fwdfit had albedo
;   component (and one of primary sources had albedo_apply on) - this plot is made from image so calculated vis don't include
;   albedo component (whereas similar plot from vis_fwdfit_plot includes albedo since has to compare to observed vis)
; 31-Oct-2017, Kim. Check that vf_srcin is a structure before using (for VIS_FWDFIT, when checking for albedo)
; 24-Jan-2018, Kim. Added Front or Rear to X title (and changed 'Subcollimator' to 'Detector')
;   Also, compute yrange from last 20 elements of vis from image (as well as last 20 elements of obs vis)
; 14-Mar-2018, Kim. If plot dev is already PS, then don't open or close PS file, just adding to a PS file.
; 19-Apr-2018, Kim. Fixed bug (thanks Jana K.) where dets_list index could be -1 (vis contains only the detectors that are ON)
; 08-May-2018, Kim. Added jpeg option. Changed file_ps keyword to file_plot.  Previously setting file_ps enabled PS mode, now it
;   doesn't.  Changed dir_ps keyword to dir_plot.
; 31-July-2018, Kim. For CLEAN images, remove residuals (only possible if clean_regress_combine = 'full_resid' or 'disable')
;   (since vis_map2vis shouldn't ever deal with negative values in images). Changed input arg to image_in to protect it from changes.
;   Changed order of plotting so that symbols will be on top of blue error lines. Changed character sizing, and made jpeg plot lines
;   thicker.
; 25-sep-2018, Kim. Added quiet keyword. Made /quiet the default to vis getdata calls. Added this_det_index_mask to calls to getdata
;   to get the right detectors (something else changed to make this necessary, not sure what). Set clean_component map to all zeroes
;   of correct dimension if image didn't succeed.
; 19-Nov-2018, Kim. Added checks for image all zero or visin not a structure so we can abort instead of crash.
; 30-Apr-2019, Kim. Added no_plot keyword
; 21-May-2019, Kim. Separated the code that compares the visibilities into hsi_image_vis_chi2.  Now we call
;   that routine to get the vis values to plot and chisquare values of comparisons, and this routine just handles plotting.
;   Also, moved the red lines that show which detectors were used in a little so they show up better.
;-

pro hsi_image_vis_plot_1, det_index_mask=det_index_mask, image=image_in, $
  visfile=visfile, jpeg=jpeg, file_plot=file_plot
  ;
  ;  checkvar, tb_index, 0
  ;  checkvar, eb_index, 0
  ;  checkvar, quiet, 1
  ;
  ;  checkvar, pwindow, imobj->get(/viscomp_window)
  ;  checkvar, ps, imobj->get(/viscomp_ps_plot)
  ;  checkvar, jpeg, imobj->get(/viscomp_jpeg_plot)
  ;  checkvar, dir_plot, imobj->get(/viscomp_plot_dir)

  ; chivals = hsi_image_vis_chi2(imobj=imobj, image=image_in, $
  ;    tb_index = tb_index, eb_index = eb_index, last=last, force=force, $
  ;    visfile=visfile, visobj=visobj, $
  ;    visvals=visvals, $
  ;    status=status, $
  ;    quiet=quiet, $
  ;    _extra=_extra)


  ;  if status eq 0 then return
  ;
  ;  if keyword_set(no_plot) then return

  ;    alg = hsi_get_alg_name(imobj->get(/image_algorithm))
  ;    alg_prefix = (hsi_alg_units(alg)).prefix
  ;    if alg_prefix eq '' then alg_prefix = 'bproj_image'

  ;   add_to_ps = !d.name eq 'PS'
  ;   ps = keyword_set(ps) or add_to_ps
  jpeg = 1; ~ps and keyword_set(jpeg)

  tvlct, rr,gg,bb, /get
  thisdevice = !d.name
  linecolors, /quiet
  charsize = 1.
  leg_size=1.
  thick = 1
  bw = 255

  ;    if ps or jpeg then begin
  if jpeg then begin

    ;      charsize = .8
    ;      leg_size = .8

    charsize = 1.6
    leg_size = 1.4
    ;      if ~is_string(file_plot) then begin
    ;        t0 = time2file(visvals.tbins[0,tb_index],/sec)
    ;        e0=trim(visvals.ebins[0,eb_index])
    ;        e1=trim(visvals.ebins[1,eb_index])
    ;        file_plot = 'viscomp_'+alg_prefix+'_'+t0+'_'+e0+'_'+e1+'kev.' + (ps ? 'ps' : 'jpeg')
    ;      endif
    ;      if is_string(dir_plot) then begin
    ;        if ~is_dir(dir_plot) then file_mkdir, dir_plot
    ;        full_file_plot = concat_dir(dir_plot, file_plot)
    ;      endif else full_file_plot = file_plot
    ;      if ps then begin
    ;        if ~add_to_ps then ps, full_file_plot, /land,/color
    ;        thick = 3.
    ;        bw = 0
    ;      endif
    full_file_plot = file_plot


    if jpeg then begin
      xsize=750
      ysize=500
      thick = 1.6
      set_plot, 'z'
      device, set_resolution=[xsize, ysize]
      temp = !P.Color
      !P.Color = !P.Background
      !P.Background = temp
    endif

  endif else begin

    save_window = !d.window
    if ~is_wopen(pwindow) then begin
      pwindow = next_window(/user)
      device, get_screen_size=sc
      window, pwindow, xsize=800<.9*sc[0], ysize=500<.9*sc[1]
    endif
    wset, pwindow
    imobj->set, viscomp_window=pwindow

  endelse

  title = '';'Visibilities - Observed, From Image, and Differences'
  units = 'photons cm!U-2!N s!U-1!n'
  xtitle = 'Front Detector + Position Angle / 180.'

  ; If yrange not passed in, compute max of last 20 elements of obs. vis. amplitudes to avoid scaling to bad
  ; values on finer collimators
  if ~keyword_set(yrange) then begin
    s = sort(visfile.isc)
    ampobsa = abs(visfile.obsvis)
    vismap = vis_map2vis(image_in, dummy, visfile)
    ampobsmapa = abs(vismap.obsvis)
    yrange = [ 0., max([ampobsa[last_nelem(s,20)], ampobsmapa[last_nelem(s,20)]]) ]
  endif
  ; yrange = [0.,40]

  udet = where(det_index_mask)
  dummy = hsi_vis_select(visfile, PAOUT=paout)       ; calculate paout
  maxpa = max(paout) gt 180. ? 360. : 180.
  scpa = visfile.isc+1 + (paout/maxpa MOD 1)

  dummy = hsi_vis_select(vismap, PAOUT=paoutmap)       ; calculate position angle
  scpamap = vismap.isc+1 + (paoutmap/maxpa MOD 1)

  visx = FLOAT(visfile.obsvis)
  visy = IMAGINARY(visfile.obsvis)
  vismapx = FLOAT(vismap.obsvis)
  vismapy = IMAGINARY(vismap.obsvis)

  visdiff = SQRT((visx-vismapx)^2 + (visy-vismapy)^2)
  nvis = n_elements(vismapx)
  print, total( f_div(((vismapx-visx)^2 + (vismapy-visy)^2), visfile.sigamp^2 )) / (2*nvis-1)
  ;
  plot,  scpa, ampobsa, /nodata, xrange=[1,10], /xst, xtickinterval=1, xminor=-1, $
    xtitle=xtitle, ytitle=units, yrange=yrange, charsize=charsize, charthick = 3., thick = 3., _extra=_extra


  ; draw vertical dotted lines at each detector boundary
  for i=2,9 do oplot, i+[0,0], !y.crange, linestyle=1
  ; draw red box around each detector used in image
  for i=0,n_elements(udet)-1 do oplot, udet[i]+1 + [0,1,1,0,0] < 9.99 > 1.001, $
    [!y.crange[0]*[.001,.001],!y.crange[1]*[.999,.999],!y.crange[0]], col=2, thick=thick
  ; draw error bars on each observed vis amplitude
  errplot, scpa, (ampobsa-visfile.sigamp > !y.crange[0]), ampobsa+visfile.sigamp < !y.crange[1], $
    width=0, COLOR=7, thick=thick
  oplot, scpa, ampobsa, psym=7, thick=thick
  oplot, scpamap, ampobsmapa, psym=4, col=2, thick=thick
  oplot, scpa, visdiff, psym=5, symsize=1, thick=thick*2, color=10
  ;    atimes = format_intervals(minmax(visfile.trange),/ut)
  ;    aen = format_intervals(trim(minmax(visfile.erange),'(f12.2)')) + ' keV'
  ;alg_label = alg + ' ' + visvals.vis_type + ' vis,   Normalize ' + (['off','on'])[visvals.vis_normalize] + visvals.text_info
  ;chia = 'Red chi2 all dets used in image = ' + trim(chivals.chisq_detused, '(f12.2)') + '  Dets = ' + arr2str(trim(udet+1),',')
  ;chib = 'Red chi2 all dets = ' + trim(chivals.chisq_alldet, '(f12.2)') + ',  Each det = ' + arr2str(trim(chivals.chisq_detsep,'(f12.2)'),', ')
  ;    leg_text = [atimes, aen, alg_label, chia, chib, 'Observed', 'Error on Observed', 'From Image', 'Vector Difference']
  ;    leg_color = [bw,bw,bw,bw,bw,bw, 10,2,7]
  ;    leg_style = [-99, -99, -99, -99, -99, 0, 0, 0, 0]
  ;    leg_sym = [0, 0, 0, 0, 0, 7, -3, 4, 5]

  leg_text = ['Observed', 'Error on Observed', 'From Image', 'Vector Difference']
  leg_color = [bw, 10,2,7]
  leg_style = [0, 0, 0, 0]
  leg_sym = [7, -3, 4, 5]
  ;ssw_legend, leg_text, psym=leg_sym, color=leg_color, linest=leg_style, box=0, charsize=leg_size, thick=thick

  ;timestamp, /bottom, charsize=charsize*.9

  thisimage = tvrd()
  tvlct, r, g, b, /get
  image24 = bytarr(3, xsize, ysize)
  image24(0,*,*) = r(thisimage)
  image24(1,*,*) = g(thisimage)
  image24(2,*,*) = b(thisimage)
  write_jpeg, full_file_plot, image24, true=1
  set_plot, thisdevice

  ;  case 1 of
  ;    ps: if ~add_to_ps then psclose
  ;    jpeg: begin
  ;      thisimage = tvrd()
  ;      tvlct, r, g, b, /get
  ;      image24 = bytarr(3, xsize, ysize)
  ;      image24(0,*,*) = r(thisimage)
  ;      image24(1,*,*) = g(thisimage)
  ;      image24(2,*,*) = b(thisimage)
  ;      write_jpeg, full_file_plot, image24, true=1
  ;      set_plot, thisdevice
  ;    end
  ;    else: wset,save_window
  ;  endcase

  tvlct, rr,gg,bb

end