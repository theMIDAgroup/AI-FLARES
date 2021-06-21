FUNCTION amp_fwdfit_func, xx, extra = extra

  type = extra.type
  ampobs = extra.ampobs 
  sigamp = extra.sigamp 
  u = extra.u 
  v = extra.v
  n_free = extra.n_free

  n_particles = (size(xx,/dimension))[0]
  n_amp = n_elements(u)
  u = reform(u, [1, n_amp])
  v = reform(v, [1, n_amp])
  
  if type eq 'circle' then begin
    
    flux = xx[*, 0]
    ones = fltarr(1, n_amp) + 1.
    flux = reform(flux, [n_particles,1])
    flux = flux # ones
    
    fwhm = reform(xx[*,1], [n_particles,1])

    amppred = flux * exp(-(!pi^2. * fwhm^2. / (4.*alog(2.)))#(u^2. + v^2.))
    

  endif
  
  if type eq 'ellipse' then begin
  
  flux = xx[*, 0]
  ones = fltarr(1, n_amp) + 1.
  flux = reform(flux, [n_particles,1])
  flux = flux # ones
    
  fwhm = reform(xx[*,1], [n_particles,1])
  eccos = reform(xx[*,2], [n_particles,1])
  ecsin = reform(xx[*,3], [n_particles,1])

  ecmsr = sqrt( eccos^2. + ecsin^2. )
  eccen = sqrt(1. - exp(-2. * ecmsr))

  fwhmminor = fwhm * (1 - eccen^2.)^0.25
  fwhmmajor = fwhm / (1 - eccen^2.)^0.25
  
  fwhmminor = fwhmminor # ones
  fwhmmajor = fwhmmajor # ones

  pa = fltarr(size(eccen, /dim))
  ind = where(eccen gt 0.001)
  pa[ind] = atan(ecsin[ind], eccos[ind]) * !radeg + 90
  pa = reform(pa, [n_particles,1])
  

  u1 = cos(pa * !dtor) # u + sin(pa * !dtor) # v
  v1 = -sin(pa * !dtor) # u + cos(pa * !dtor) # v

  amppred = flux * exp(- !pi^2. / (4. * alog(2.)) * ((u1 * fwhmmajor)^2. + (v1 * fwhmminor)^2.))
  
  endif
  
  
  if type eq 'multi' then begin
    
  d = reform(xx[*,4], [n_particles,1])
  flux1 = xx[*,1]
  fwhm1 = reform(xx[*,0], [n_particles,1])

  flux2 = xx[*,3]
  ecc = reform(xx[*,2], [n_particles,1])
  pa = reform(xx[*,5], [n_particles,1])
  
  ones = fltarr(1, n_amp) + 1.
  flux1 = reform(flux1, [n_particles,1])
  flux2 = reform(flux2, [n_particles,1])
  flux1 = flux1 # ones
  flux2 = flux2 # ones
  

  phase = 2. * !pi * ((d * cos(pa/180.*!pi)) # u + (d * sin(pa/180.*!pi)) # v)

  re_obs1 = flux1 * exp(-(!pi^2. * fwhm1^2. / (4.*alog(2.)))#(u^2. + v^2.))*cos(phase);
  im_obs1 = flux1 * exp(-(!pi^2. * fwhm1^2. / (4.*alog(2.)))#(u^2. + v^2.))*sin(phase);

  re_obs2 = flux2 * exp(-(!pi^2. * (fwhm1 * ecc) ^2. / (4.* alog(2.)))#(u^2. + v^2.))*cos(phase);
  im_obs2 = -flux2 * exp(-(!pi^2. * (fwhm1 * ecc) ^2. / (4.* alog(2.)))#(u^2. + v^2.))*sin(phase);

  amppred = sqrt( (re_obs1 + re_obs2)^2. + (im_obs1 + im_obs2)^2. )
  
  endif
  
  if type eq 'multi_fl' then begin
    
    fwhm1 = reform(xx[*,0], [n_particles,1])
    fwhm2 = reform(xx[*,3], [n_particles,1])
    
    flux1 = xx[*,1]
    flux2 = xx[*,1] * xx[*,2]
    
    d = reform(xx[*,4], [n_particles,1])
    pa = reform(xx[*,5], [n_particles,1])
    

    ones = fltarr(1, n_amp) + 1.
    flux1 = reform(flux1, [n_particles,1])
    flux2 = reform(flux2, [n_particles,1])
    flux1 = flux1 # ones
    flux2 = flux2 # ones


    phase = 2. * !pi * ((d * cos(pa/180.*!pi)) # u + (d * sin(pa/180.*!pi)) # v)

    re_obs1 = flux1 * exp(-(!pi^2. * fwhm1^2. / (4.*alog(2.)))#(u^2. + v^2.))*cos(phase);
    im_obs1 = flux1 * exp(-(!pi^2. * fwhm1^2. / (4.*alog(2.)))#(u^2. + v^2.))*sin(phase);

    re_obs2 = flux2 * exp(-(!pi^2. *  fwhm2^2. / (4.* alog(2.)))#(u^2. + v^2.))*cos(phase);
    im_obs2 = -flux2 * exp(-(!pi^2. * fwhm2^2. / (4.* alog(2.)))#(u^2. + v^2.))*sin(phase);

    amppred = sqrt( (re_obs1 + re_obs2)^2. + (im_obs1 + im_obs2)^2. )

  endif

  chi = total((amppred - ampobs)^2./sigamp^2., 2)/n_free

  RETURN, chi
  
END

