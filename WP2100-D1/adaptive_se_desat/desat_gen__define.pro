function desat_gen::find_file, path, wav, range = valid_range
  ;+
  ; NAME:
  ; find_file
  ; PURPOSE:
  ; search all the .fts files in the path folder associated to wav wavelength.
  ; EXPLANATION:
  ; This routine will find all the .fts files inside the path folder. If will return
  ; all the files associated to the wav wavelength.
  ; CALLING SEQUENCE:
  ; result = obj -> find_file(path, '131')
  ; INPUTS:
  ; pix = string containing the path to the folder containing the aia .fts files
  ; wav = string array or value for the data wavelength to search
  ;-
  file_name = file_search(concat_dir(path,'*.f*ts'), COUNT=ct)
  if ct eq 0 then file_name = file_search(concat_dir(path,'*.fits'), COUNT=ct)
  mreadfits, file_name, info ;, data ;ras, remove data because you only want to read the headers here

  ; check the wavelength
  ind   = where(info.WAVELNTH eq round(float(wav))) ;GT added round(), 5-oct-2017
  info  = info[ ind ]
  file_name = file_name[ ind ]

  ;check the time-range
  ;ind  = where_within( anytim( info.date_obs ), valid_range )
  ind = where(anytim( info.date_obs ) ge valid_range[0] and anytim( info.date_obs ) le valid_range[1])
  info  = info[ ind ]
  file_name = file_name[ ind ]

  return,file_name
end


pro desat_gen::readfts, file
  ;+
  ; NAME:
  ; desat::readfts
  ; PURPOSE:
  ; read the .fts files and fill the object.
  ; EXPLANATION:
  ; Read the data contained in file filling the object with data and infos
  ; CALLING SEQUENCE:
  ; obj -> readfts, file
  ; INPUTS:
  ; file = string or array of string containing the file names to be read.
  ; CALLS:
  ; deconv
  ; PROCEDURE:
  ; it is used to read the data from .fts files, by means mreadfits routine, and fill
  ; the 'desat' object.
  ;-

  mreadfits, file, info, data
  index2map, info, data, map
  ;; SG, Dec-2019, check on exptime:
  ; compare the commanded exposure times in the keyword AIMGSHCE (in msec) with the measured exposure times in the keyword EXPTIME (in sec) and replace the
  ; uncorrected exposure times in EXPTIME with the ones in AIMGSHCE (converted in sec)
  indexes_uncorrected_exptime = where(abs(info.EXPTIME-(info.aimgshce*1e-3))/(info.aimgshce*1e-3) gt 0.01, count_uncorrected_exptime)
  if count_uncorrected_exptime gt 0 then begin
    info[indexes_uncorrected_exptime].exptime=info[indexes_uncorrected_exptime].aimgshce*1e-3
  endif
  self-> set, index=info, map=map, /no_copy

end


pro desat_gen::get_data_and_info, str, data, info
  ;+
  ; NAME:
  ; desat::get_data_and_info
  ; PURPOSE:
  ; returns the data, background and info array for the selected time interval for
  ; saturated frames.
  ; EXPLANATION:
  ; it will return the data array for the saturated images inside the selected time
  ; interval.
  ; At the same time this routine will return a computed estimation for data
  ; background by means the dst_bkg_fourier_interp.pro routine. Moreover, the unsaturated
  ; images inside the selected time interval are deconvolved with the
  ; Expectation Maximization algorithm.
  ; CALLING SEQUENCE:
  ; obj -> get_data_and_info, str, data, bg, info
  ; INPUTS:
  ; str  = dst_str global structure for the desaturation routine.
  ; OUTPUT:
  ; data = data 3D corresponding to saturated images into the selected time interval
  ; bg   = computed background estimation obtained by dst_bkg_fourier_interp routine
  ; info = index structure associated to data array
  ; CALLS:
  ; CALLED BY:
  ;   DESATURATION
  ; CALLS:
  ;   DECONV
  ;   dst_bkg_fourier_interp
  ; PROCEDURE:
  ;-

  ;;; cut data array to be npix*npix pixel
  self ->ctrl_data, str.npix

  ;; data/info array storage
  ;; GT, 5-oct-2017, Edited and improved this message for clarity
  if max(*str.sat_ind) eq -1 then begin
    print, ''
    print, '***************************************************'
    print, '* No saturated images for the selected time range *'
    print, '***************************************************'
    print, ''
    message, ''

  endif else begin
    info  = (self ->get(/index))[*str.sat_ind]
    data    = ((self ->get(/data))>0)[*,*,*str.sat_ind]
  endelse
end


pro desat_gen::indices_analysis, str, aec = aec
  ;+
  ; NAME:
  ; desat::indices_analysis
  ; PURPOSE:
  ; It will return indices for the considering event for saturated, background in the
  ; considered time interval.
  ; EXPLANATION:
  ; Analyzing the infos contained into the index structure it is possible to retrive
  ; the indices for saturated frames, the time frames that will be used for the background
  ; estimation step.
  ; Moreover, are defined the indices the selected time interval.
  ; CALLING SEQUENCE:
  ; obj -> indices_analysis, str, aec = aec
  ; INPUTS:
  ; str  = dst_str global structure for the desaturation routine.
  ; OUTPUT:
  ; str.time_ind = selected time interval indices
  ; str.sat_ind  = saturated frames indices
  ; str.bg_ind   = indices for the frame that will be used into the background estimation step
  ; KEYWORDS:
  ; aec = if set also saturated images with short exp. time were desaturated
  ; CALLS:
  ; CALLED BY:
  ;   DESATURATION
  ;-

  default, aec, 1

  info = self ->get(/index) ;done after this, we get info from reading all the headers, not the data

  ;;; file indices for the selected time interval
  q_time_range = anytim(info.date_obs) gt anytim(str.ts) and anytim(info.date_obs) lt anytim(str.te)

  ;;; saturated frame indices
  q_sat_frame = info.DATAMAX ge str.sat_lev

  ;;; short exposure time file indices

  q_short_exp = size(/n_e,info) gt 1 ? info.exptime lt 0.8*median(info.exptime) : info.exptime lt 0.7*max(info.exptime)

  ;;; indices for saturated frames in the selected time interval
  sat_ind = ~keyword_set(aec) ? where(q_time_range and q_sat_frame and ~q_short_exp) : where(q_time_range and q_sat_frame)

  ;;; background indices estimation over the whole dataset
  q_bg_dataset = total(q_short_exp) gt 0 ? ~q_sat_frame or q_short_exp : ~q_sat_frame
  bg_dataset = where(q_bg_dataset)

  ;;; background indices estimation in the select time interval
  bg_ind = where(q_time_range and q_bg_dataset)

  ;;; background indices for background estimation routine (bg_ind_t_ind-2 < bg_ind < bg_ind_t_ind+2)
  for i = 0 , 1 do begin

    prew_ind = where(bg_dataset lt min(bg_ind), ct)
    if ct gt 0 then bg_ind = [ max(bg_dataset[prew_ind]) , bg_ind ]

    post_ind = where(bg_dataset gt max(bg_ind), ct)
    if ct gt 0 then bg_ind = [ bg_ind , min(bg_dataset[post_ind]) ]

  endfor

  str.time_ind  = ptr_new(where(q_time_range))
  str.sat_ind   = ptr_new(sat_ind)
  str.bg_ind  = ptr_new(bg_ind)
end

pro desat_gen::_bld_filenames, index, data_path, peaklam = peaklam, PRiL=PRiL, model=model 
  ;+
  ; NAME:
  ; desat::_bld_filenames
  ; PURPOSE:
  ; save data array into an .fts file into defined data_path folder
  ; EXPLANATION:
  ; if original is set it will save original data
  ; if desat is set it will save original data
  ; CALLING SEQUENCE:
  ; obj -> savefts, index, data_path, original=original, desat=desat
  ; INPUTS:
  ; index  = indices of data images in the obj that has to be saved.\
  ; data_path = path of the save folder
  ; info_str = informational structure with details of deconvolution
  ; KEYWORDS:
  ; original = set if original data has to be saved (before the desaturation procedure)
  ; desat = set if desat. data ha to be saved (after the desaturation procedure)
  ; CALLS:
  ; CALLED BY:
  ;   DESATURATION
  ; CALLS:
  ;   mwritefts, mkdir, file_mkdir
  ; PROCEDURE:
  ;-

  n_el = size(/n_e,index)

  info = (self -> get(/index))[index]

  stringtime = anytim(info.date_obs,/vms,/date_only)
  stringwave = strtrim(string(peaklam),1)
  stringwave = strjoin(strsplit(stringwave,'.', /EXTRACT), 'p') ;GT, filename format modification , 5-oct-2017

  path = concat_dir(stringtime, stringwave)
  path = concat_dir(data_path, path)

  path_or  = concat_dir(path,'data')
  path_sat = concat_dir(path,'desat')

  if ~dir_exist( path_or[0] ) then file_mkdir, path_or
  if ~dir_exist( path_sat[0] ) then file_mkdir, path_sat
  ;use time2file in constructing filenames!

  orig_file = concat_dir(path_or,'aia_orig_'+time2file(info.date_obs, /sec )+'_'+stringwave+'.fts')
  desat_file = concat_dir(path_sat,'aia_desat_'+time2file(info.date_obs, /sec )+'_'+stringwave+'.fts')
  if keyword_set(PRiL) then begin
    if keyword_set(model) then begin
      desat_file = concat_dir(path_sat,'aia_PRiL_model_'+time2file(info.date_obs, /sec )+'_'+stringwave+'.fts')
    endif else begin
      desat_file = concat_dir(path_sat,'aia_PRiL_'+time2file(info.date_obs, /sec )+'_'+stringwave+'.fts')
    endelse
  endif



  filenames = replicate( {date_obs:'', orig: '', desat: ''}, n_el)
  filenames.date_obs = info.date_obs
  filenames.orig     = orig_file
  filenames.desat    = desat_file

  self.filenames = ptr_new( filenames )
end

function desat_gen::_Get_Filename, date_obs, original = original,  index = index
  filenames = *Self.filenames
  default, original, 0
  desat = 1 - original
  info = (self -> get(/index))
  data_class = desat ? 'desat' : 'orig'
  tg_id = stregex(/fold, /boo, 'desat', data_class) ? tag_index( filenames, 'DESAT' ) : tag_index( filenames, 'ORIG' )


  if exist( date_obs ) then begin
    select = where(  anytim( filenames.date_obs ) eq anytim( date_obs ), nsel )
    if nsel eq 0 then message,'date_obs does not match any of the filenames!'
    filename = filenames[select].(tg_id)
    filenames[select].(tg_id) = '' ;remove so we don't overwrite the desat image with the original at the last fits write
    *Self.filenames = filenames
  endif else begin
    ;We must want all remaining desat filenames
    filename = filenames.(tg_id)
    ;Only retrieve the remaining valid filenames
    select = where( filename ne '', nsel)

    filename = nsel ge 1 ? filename[ select ] : -1
  endelse
  date_obs_4_index = filenames[select].date_obs
  index = where_arr( info.date_obs, date_obs_4_index )

  return, filename
end


pro desat_gen::savefts, date_obs, original=original, desat = desat, use_prep=use_prep, info_str = info_str, PRiL = PRiL
  ;+
  ; NAME:
  ; desat::savefts
  ; PURPOSE:
  ; save data array into an .fts file into defined data_path folder
  ; EXPLANATION:
  ; if original is set it will save original data
  ; if desat is set it will save original data
  ; CALLING SEQUENCE:
  ; obj -> savefts, index, data_path, original=original, desat=desat
  ; INPUTS:
  ; index  = indices of data images in the obj that has to be saved.\
  ; data_path = path of the save folder
  ; info_str = informational structure with details of deconvolution
  ; KEYWORDS:
  ; original = set if original data has to be saved (before the desaturation procedure)
  ; desat = set if desat. data ha to be saved (after the desaturation procedure)
  ; CALLS:
  ; CALLED BY:
  ;   DESATURATION
  ; CALLS:
  ;   mwritefts, mkdir, file_mkdir
  ; PROCEDURE:
  ;-

  default, original, 0
  default, use_prep, 1
  desat = 1 - original

  tmp_file = Self->_Get_Filename( date_obs, original=original, index = index )
  data = (self -> get(/data))[*,*, index]
  info = (self -> get(/index))[index]
  if keyword_set( use_prep ) then begin
    aia_prep, info, data, oindex, odata, /cutout
    info = oindex
    data = odata
    Self->update_history, info, 'desat alg'

  endif

  MWRITEFITS, info, data, outfile=tmp_file
  if exist( info_str ) && is_struct( info_str ) then begin
    if keyword_set(PRiL) then begin
      info = { desat_flux: *info_str.sat_flux, $
        background: *info_str.bg, $
        sat_core: *info_str.s, $
        sat_fringe: *info_str.g, $
        sat_bloom: *info_str.b, $
        c_stat: info_str.c_stat, $
        sat_level: info_str.sat_lev }
      ;mwrfits, info, ssw_strsplit( tmp_file, '.fts' ) + '_desat_info.fts'
      
    endif else begin
      info = { desat_flux: *info_str.sat_flux, $
        background: *info_str.bg, $
        sat_core: *info_str.s, $
        sat_fringe: *info_str.g, $
        sat_bloom: *info_str.b, $
        c_stat: info_str.c_stat, $
        ex_max_target: info_str.lev, $
        sat_level: info_str.sat_lev }
      ;mwrfits, info, ssw_strsplit( tmp_file, '.fts' ) + '_desat_info.fts'
    endelse
    

    mwrfits, info, tmp_file, /silent
    fxhmodify, tmp_file,'EXTEND',1
    fxhmodify, tmp_file, 'EXTNAME  ','DESAT_INFO', /extension
  endif

  data_path = file_dirname( file_search(tmp_file[0], /full) )
  if keyword_set(original) then print, 'Original .fts file are saved in ' + data_path + ' folder'
  if keyword_set(desat) then begin
    if keyword_set(PRiL) then begin
      print, 'De-saturated .fts file are saved in  ' + data_path + ' folder'
    endif else begin
    print, 'De-saturated .fts file are saved in  ' + data_path + ' folder'
    endelse
  endif
  

end


pro desat_gen::update_history, info, records, _extra=extra
  update_history, info, 'desat alg', _extra = extra ;initial implementation of history record, ras, 21-feb-2015
  ;  update_history, index,  records,   mode=mode , debug=debug, $
  ;    caller=caller, routine=routine, noroutine=noroutine, version=version
  ;+
  ;    Name: update_history
  ;
  ;    Purpose: add history record(s) to input structure(s)
  ;
  ;    Input Parameters:
  ;      index   - structure vector or FITs header array
  ;      records - info to add - string/string array
  ;
  ;    Keyword Parameters:
  ;      routine - if set , routine name to prepend to each history record -
  ;                default is via: 'get_caller' unless /noroutine set
  ;      caller - synonym for routine
  ;      noroutine - if set, dont prepend 'Caller' to records
  ;      version   - if set, verion number, include 'VERSION:' string
  ;
  ;      mode - if set and #records=#index, add record(i)->index(i)
  ;             (default mode = record(*)->index(*) (all records->all structure)
  ;-
end

function desat_gen::dst_psf_gen, wavelength, npix, dwavelength, core_dim
  ;+
  ; NAME:
  ; dst_psf_gen
  ; PURPOSE:
  ; returns the psf structure that contains the diffraction and dispersion and the complete
  ; aia psf.
  ; EXPLANATION:
  ; The dispersion component of the psf is considered cutting out a circular portion on the
  ; central component of the psf computed by aia_calc_psf_mod.pro. The radius of this circle is
  ; fixed to 5 pixels but it can be modified by users.
  ; CALLING SEQUENCE:
  ; psf = dst_psf_gen( info , dwavelength , core_dim )
  ; INPUTS:
  ; info    = index structure for the aia data, necessary to retrieve infos about the psf
  ;       to generate
  ; dwavelength   = correction parameter on the wavelength of the psf (look aia_calc_psf_mod.pro)
  ; core_dim  = radius of the central core of the psf.
  ; OUTPUT:
  ; psf.cpsf = diffraction component of the psf
  ; psf.opsf = dispersion component of the psf
  ; psf.psf  = complete psf
  ; CALLS:
  ; CALLED BY:
  ;   DESATURATION
  ; CALLS:
  ;   aia_calc_psf_mod.pro
  ;-

  default, core_dim, 5

  psf   = fltarr(npix, npix, n_elements(dwavelength))
  cpsf  = fltarr(npix, npix, n_elements(dwavelength))
  opsf  = fltarr(npix, npix, n_elements(dwavelength))

  ; build a circular mask
  xgrid = (fltarr(npix)+1)##indgen(npix)
  ygrid = indgen(npix)##(fltarr(npix)+1)
  center  = [fix(npix/2.),fix(npix/2.)]
  w   = where((xgrid-center[0])^2+(ygrid-center[1])^2 le core_dim^2)
  mask_c  = fltarr(npix,npix) & mask_c[w] = 1

  for iw = 0 , n_elements(dwavelength)-1 do begin
    print, iw
    wav = (dwavelength[iw] * float(wavelength)) + float(wavelength)
    file = file_search(concat_dir('AIA_PSF' ,'AIA_PSF_'+ strtrim(wav,1) +'_'+strtrim(npix,1)+'.fts'), count=ct)

    if ct gt 0 then begin
      mreadfits, file, info, psf_IN
    endif else begin
      FILE_MKDIR , 'AIA_PSF'
      psf_in = aia_calc_psf_mod( wavelength, npix = npix, dwavelength = dwavelength[iw])
      m1    = fltarr(npix,npix) & m1[npix/2 , npix/2 ] = 1
      psf_in  = convolve(/corr, m1, psf_in) > 0.0
      mwrfits, psf_in , concat_dir('AIA_PSF' ,'AIA_PSF_'+ strtrim(wav,1) +'_'+strtrim(npix,1)+'.fts')
    endelse

    psf[*,*,iw]  = psf_in
    opsf[*,*,iw] = REFORM(psf[*,*,iw] * mask_c)
    cpsf[*,*,iw] = REFORM(psf[*,*,iw] * (1 - mask_c))

  endfor
  psf   = {wavelength: wavelength, dwavelength: dwavelength, npix: npix, cpsf:cpsf, opsf:opsf, psf:psf}
  return, psf
end


function desat_gen::psf, info, dwavelength, core_dim
  ;+
  ; NAME:
  ; desat::psf
  ; PURPOSE:
  ; returns the psf structure that contains the diffraction and dispersion and the complete
  ; aia psf.
  ; EXPLANATION:
  ; The dispersion component of the psf is considered cutting out a circular portion on the
  ; central component of the psf computed by aia_calc_psf_mod.pro. The radius of this circle is
  ; fixed to 5 pixels but it can be modified by users.
  ; CALLING SEQUENCE:
  ; psf = self->psf( info , dwavelength , core_dim )
  ; INPUTS:
  ; info    = index structure for the aia data, necessary to retrieve infos about the psf
  ;       to generate
  ; dwavelength   = correction parameter on the wavelength of the psf (look aia_calc_psf_mod.pro)
  ; core_dim  = radius of the central core of the psf.
  ; OUTPUT:
  ; psf.cpsf = diffraction component of the psf
  ; psf.opsf = dispersion component of the psf
  ; psf.psf  = complete psf
  ; CALLS:
  ; CALLED BY:
  ;   DESATURATION
  ; CALLS:
  ;   aia_calc_psf_mod.pro
  ;-

  npix = min([info.naxis1,info.naxis2])
  wavelength = strtrim( string(info[0].WAVELNTH),1)
  have_psf = 0

  if ptr_valid( Self.psf ) then begin
    psf = *Self.psf
    have_psf = npix eq psf.npix && wavelength eq psf.wavelength && ARRAY_EQUAL(dwavelength, psf.dwavelength) eq 1;dwavelength eq psf.dwavelength
  endif

  if ~have_psf then begin
    default, core_dim, 5
    default, dwavelength, 0.0
    psf = self->dst_psf_gen( wavelength, npix, dwavelength, core_dim )
    self.psf = ptr_new( psf )
  endif


  return, psf

end

pro desat_gen::ctrl_data, pix, data = data_, info = info_
  ;+
  ; NAME:
  ; desat::ctrl_data
  ; PURPOSE:
  ; Reduce the FOV of a given data array.
  ; EXPLANATION:
  ; Reduce the dimension of the FOV of a given data array to [pixel,pixel].
  ;   If data and info are given, the routine vill redice them, otherwise it will act on
  ; the object data and info
  ; CALLING SEQUENCE:
  ; ctrl_data, pix                    acting on the object
  ; ctrl_data, pix, data = data_in, info = info_in  acting on data_in and info_in
  ; INPUTS:
  ; pix = number of pixel for the new FOV
  ; OPTIONAL:
  ; data_ = 3D data array on wich we what to act
  ; info_ = index array structure for data_ array
  ; KEYWORDS:
  ; CALLS:
  ; data_restore_2
  ; PROCEDURE:
  ; uses the extract_slice function to extract the new FOV around the center of the
  ; of the image. The new resized data are store in the previous object refreshing
  ; the infos array.
  ;-
  var = exist(data_)+exist(info_) eq 2 ? 1 : 0

  default, data, self -> get(/data)
  default, info, self -> get(/index)

  if var eq 1 then data = data_
  if var eq 1 then info = info_

  n_f = size(/n_e, info)

  img_cent = [(size(/dim, data))[0:1]]/2

  data_rescaled = n_f eq 1 ? fltarr(pix, pix) : fltarr(pix, pix, n_f)

  for i = 0, n_f - 1 do begin

    zi = i
    if n_f eq 1 then begin
      data_rescaled[*,*] = EXTRACT_SLICE( [[[data]],[[data]]], pix, pix, img_cent[0], img_cent[1], zi, 0, 0, 0 )
    endif else begin
      data_rescaled[*,*,i] = EXTRACT_SLICE( data, pix, pix, img_cent[0], img_cent[1], zi, 0, 0, 0 )
    endelse

    xp=(mk_map_xp(info[i].xcen,info[i].cdelt1,info[i].naxis1,1))[img_cent[0]]
    yp=(mk_map_yp(info[i].ycen,info[i].cdelt2,1,info[i].naxis1))[img_cent[1]]

    info[i].xcen = xp
    info[i].ycen = yp

    info[i].naxis1 = pix
    info[i].naxis2 = pix

    info[i].crpix1 = info[i].crpix1 - img_cent[0] + (pix/2.)
    info[i].crpix2 = info[i].crpix2 - img_cent[1] + (pix/2.)

  endfor

  if var eq 0 then begin
    index2map, info, data_rescaled, map
    self-> set, index=info, map=map
  endif else begin
    data_ = data_rescaled
    info_ = info
  endelse
end




pro desat_gen__define, void

  void={desat_gen, $
    filenames: ptr_new(), $
    psf: ptr_new(), $
    inherits sdo}
  return

end
